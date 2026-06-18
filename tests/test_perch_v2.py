"""TF-free tests for the Perch 2 adapter (``faun.classification.perch_v2``).

These tests never touch real TensorFlow, kagglehub, or the network. Inference is
exercised by injecting a *fake* SavedModel whose ``serving_default`` signature is
a plain Python callable returning numpy arrays, so we validate the adapter's
preprocessing + ranking + dim contract without the heavy stack.

The creds/honesty gate is a MERGE-GATE: with no model path and no Kaggle creds,
``Perch2Adapter()`` must hard-fail BEFORE any network access and must NEVER fall
back to Perch 1 (that would corrupt the embedding dim/provenance).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from faun.classification import Prediction
from faun.classification.perch_v2 import PERCH_V2_DIM, Perch2Adapter


# ---------------------------------------------------------------------------
# Fake SavedModel: a stand-in for tf.saved_model.load(path).
# ---------------------------------------------------------------------------


class _FakeSignature:
    """Mimics ``model.signatures['serving_default']`` for Perch 2.

    Records the last ``inputs`` it was called with so tests can assert the
    preprocessed shape, and returns a dict shaped like the real Perch 2 output.
    """

    def __init__(self, dim: int = PERCH_V2_DIM, n_classes: int = 3) -> None:
        self.dim = dim
        self.n_classes = n_classes
        self.last_inputs: np.ndarray | None = None

    def __call__(self, inputs):
        self.last_inputs = np.asarray(inputs)
        n = self.last_inputs.shape[0]
        # Deterministic logits so the top-1 is predictable: class 1 wins.
        logits = np.tile(np.array([0.1, 0.9, 0.5], dtype=np.float32), (n, 1))
        return {
            "embedding": np.arange(self.dim, dtype=np.float32)[np.newaxis, :].repeat(
                n, axis=0
            ),
            "spatial_embedding": np.zeros((n, 5, 3, self.dim), dtype=np.float32),
            "label": logits[:, : self.n_classes],
            "spectrogram": np.zeros((n, 4, 4), dtype=np.float32),
        }


class _FakeModel:
    """Stand-in for a loaded SavedModel exposing ``.signatures``."""

    def __init__(self, signature: _FakeSignature) -> None:
        self.signatures = {"serving_default": signature}


def _adapter_with_fake(monkeypatch, signature: _FakeSignature) -> Perch2Adapter:
    """Build an adapter past the creds gate with a fake model injected."""
    # A model_path satisfies the source-resolution gate (no creds needed).
    adapter = Perch2Adapter(model_path="/fake/perch_v2", labels=["a", "b", "c"])
    # Inject the fake so _load() is never reached / TF never imported.
    adapter._model = _FakeModel(signature)
    return adapter


# ---------------------------------------------------------------------------
# Honesty / creds merge-gate
# ---------------------------------------------------------------------------


def _strip_kaggle_env(monkeypatch, tmp_path):
    """Remove every Kaggle credential source and point HOME at an empty dir."""
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("PERCH_V2_MODEL_PATH", raising=False)
    # Redirect HOME so ~/.kaggle/kaggle.json cannot exist.
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setattr("pathlib.Path.home", lambda: empty_home)


def test_no_creds_no_path_raises_runtimeerror(monkeypatch, tmp_path):
    """MERGE-GATE: no model_path + no Kaggle creds -> explicit RuntimeError."""
    _strip_kaggle_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        Perch2Adapter()
    msg = str(excinfo.value)
    # Message must name Perch 2 and the missing creds (honest diagnosis).
    assert "Perch 2" in msg or "Perch2" in msg
    assert "kaggle" in msg.lower() or "cred" in msg.lower()


def test_no_creds_does_not_fall_back_to_perch1(monkeypatch, tmp_path):
    """The gate must NOT silently construct/return a Perch 1 adapter."""
    _strip_kaggle_env(monkeypatch, tmp_path)
    # Poison: if perch_v2 ever imported PerchAdapter, this would be a red flag.
    sentinel = {"called": False}

    import faun.classification as classification

    real_getattr = classification.__getattr__

    def spy_getattr(name):
        if name == "PerchAdapter":
            sentinel["called"] = True
        return real_getattr(name)

    monkeypatch.setattr(classification, "__getattr__", spy_getattr)
    with pytest.raises(RuntimeError):
        Perch2Adapter()
    assert sentinel["called"] is False, "Perch 2 must never fall back to Perch 1"


def test_kaggle_creds_env_passes_gate_lazily(monkeypatch, tmp_path):
    """With KAGGLE_USERNAME+KAGGLE_KEY set, construction must NOT hit network.

    The gate is satisfied by creds; the actual download is deferred to _load().
    """
    _strip_kaggle_env(monkeypatch, tmp_path)
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "deadbeef")
    # Constructor must not import kagglehub or TF.
    adapter = Perch2Adapter()
    assert adapter._model is None  # nothing loaded yet
    assert "kagglehub" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_kaggle_json_file_passes_gate(monkeypatch, tmp_path):
    """A ~/.kaggle/kaggle.json file satisfies the creds gate."""
    _strip_kaggle_env(monkeypatch, tmp_path)
    home = tmp_path / "with_kaggle"
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    adapter = Perch2Adapter()  # must not raise
    assert adapter._model is None


# ---------------------------------------------------------------------------
# Dim guard
# ---------------------------------------------------------------------------


def test_dim_constant_is_1536():
    """Perch 2 embedding dim is 1536 (NOT Perch-1's 1280)."""
    assert PERCH_V2_DIM == 1536


def test_adapter_advertises_1536_not_1280(monkeypatch):
    """The adapter exposes DIM=1536 so a 1280-vs-1536 mismatch is detectable."""
    adapter = Perch2Adapter(model_path="/fake/perch_v2")
    assert adapter.DIM == 1536
    assert adapter.DIM != 1280


def test_embed_returns_1536_vector(monkeypatch):
    """embed() returns a flat (1536,) vector from the model embedding output."""
    sig = _FakeSignature(dim=PERCH_V2_DIM)
    adapter = _adapter_with_fake(monkeypatch, sig)
    wav = np.zeros(32_000, dtype=np.float32)  # 1 s @ 32k
    emb = adapter.embed(wav, sr=32_000)
    assert emb.shape == (1536,)
    assert emb.dtype == np.float32


def test_embed_mismatched_dim_is_detectable(monkeypatch):
    """If the model returns 1280-dim embeddings, embed() surfaces the wrong shape.

    The adapter does not silently coerce; a 1280-vs-1536 mismatch is visible to
    callers (the provenance/dim contract stays honest).
    """
    sig = _FakeSignature(dim=1280)
    adapter = _adapter_with_fake(monkeypatch, sig)
    emb = adapter.embed(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert emb.shape == (1280,)
    assert emb.shape != (adapter.DIM,)


# ---------------------------------------------------------------------------
# classify() ranking
# ---------------------------------------------------------------------------


def test_classify_returns_ranked_predictions(monkeypatch):
    """classify() returns a ranked list[Prediction] using the 'label' logits."""
    sig = _FakeSignature()
    adapter = _adapter_with_fake(monkeypatch, sig)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert isinstance(preds, list)
    assert all(isinstance(p, Prediction) for p in preds)
    # Fake logits [0.1, 0.9, 0.5] -> labels ["a","b","c"] -> top is "b".
    assert preds[0].species == "b"
    # Descending order by probability.
    probs = [p.probability for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_classify_respects_top_k(monkeypatch):
    """top_k caps the number of returned predictions."""
    sig = _FakeSignature(n_classes=3)
    adapter = Perch2Adapter(model_path="/fake", labels=["a", "b", "c"], top_k=2)
    adapter._model = _FakeModel(sig)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert len(preds) == 2


def test_classify_default_labels_when_absent(monkeypatch):
    """Without explicit labels, predictions are named ``species_<i>``."""
    sig = _FakeSignature(n_classes=3)
    adapter = Perch2Adapter(model_path="/fake")  # no labels
    adapter._model = _FakeModel(sig)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].species.startswith("species_")


# ---------------------------------------------------------------------------
# Preprocessing: resample + pad/crop to exactly 160000 @ 32k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sr", [16_000, 48_000, 32_000])
def test_preprocess_pads_to_160000_at_32k(monkeypatch, sr):
    """16k/48k/32k mono input is resampled+fit to exactly (1, 160000)."""
    sig = _FakeSignature()
    adapter = _adapter_with_fake(monkeypatch, sig)
    # 2 s of audio at each sr -> after resample to 32k, ~64000 samples, padded.
    wav = np.zeros(2 * sr, dtype=np.float32)
    adapter.classify(wav, sr=sr)
    assert sig.last_inputs is not None
    assert sig.last_inputs.shape == (1, 160_000)
    assert sig.last_inputs.dtype == np.float32


def test_preprocess_crops_long_input(monkeypatch):
    """Input longer than 5 s @ 32k is cropped to exactly 160000."""
    sig = _FakeSignature()
    adapter = _adapter_with_fake(monkeypatch, sig)
    wav = np.zeros(10 * 32_000, dtype=np.float32)  # 10 s
    adapter.embed(wav, sr=32_000)
    assert sig.last_inputs.shape == (1, 160_000)


def test_preprocess_downmixes_stereo(monkeypatch):
    """Stereo (frames, 2) input is downmixed to mono before windowing."""
    sig = _FakeSignature()
    adapter = _adapter_with_fake(monkeypatch, sig)
    stereo = np.ones((32_000, 2), dtype=np.float32)
    adapter.embed(stereo, sr=32_000)
    assert sig.last_inputs.shape == (1, 160_000)


# ---------------------------------------------------------------------------
# Import is TF-free
# ---------------------------------------------------------------------------


def test_module_import_is_tf_free():
    """Importing the module must never pull TensorFlow or kagglehub."""
    # Force a fresh import to make the assertion meaningful even if a sibling
    # test already imported the module.
    sys.modules.pop("faun.classification.perch_v2", None)
    import faun.classification.perch_v2  # noqa: F401

    assert "tensorflow" not in sys.modules
    assert "kagglehub" not in sys.modules


def test_load_is_lazy(monkeypatch):
    """_load is only invoked at first inference, not at construction."""
    adapter = Perch2Adapter(model_path="/fake/perch_v2")
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        adapter._model = _FakeModel(_FakeSignature())
        return adapter._model

    monkeypatch.setattr(adapter, "_load", fake_load)
    assert calls["n"] == 0  # construction did not load
    adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert calls["n"] == 1  # first call loaded once
    adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert calls["n"] == 2  # _load is called each inference (cached inside)
