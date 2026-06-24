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
from faun.classification.perch_v2 import (
    PERCH_V2_DIM,
    Perch2Adapter,
    _softmax,
    apply_presence_gate,
    bird_presence_mass,
)
from faun.settings import get_settings


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


# ===========================================================================
# Wave C — presence soft-gate (FR-003) + serve-time calibration (FR-006-serve)
# ===========================================================================


# ----- free-function math (SC-C3), no TF -----------------------------------


def test_softmax_is_a_distribution() -> None:
    p = _softmax(np.array([2.0, 1.0, -3.0, 0.5]))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p >= 0.0)
    # monotone in the logits.
    assert np.argmax(p) == 0


def test_bird_presence_mass_sums_bird_columns() -> None:
    scores = np.array([2.0, 1.0, -3.0, 0.5])
    mask = np.array([True, False, True, False])
    probs = _softmax(scores)
    assert bird_presence_mass(scores, mask) == pytest.approx(float(probs[0] + probs[2]))
    # All-bird -> mass 1.0; no-bird -> 0.0.
    assert bird_presence_mass(scores, np.ones(4, bool)) == pytest.approx(1.0)
    assert bird_presence_mass(scores, np.zeros(4, bool)) == pytest.approx(0.0)


def test_apply_presence_gate_boost_and_clamp() -> None:
    # k=0 is the identity (the adapter never calls it at k=0, but the math holds).
    assert apply_presence_gate(0.4, 0.7, 0.0) == pytest.approx(0.4)
    # k>0 with bird mass boosts the species probability.
    assert apply_presence_gate(0.4, 0.5, 2.0) == pytest.approx(0.4 * (1 + 0.5 * 2))
    # clamped into [0, 1].
    assert apply_presence_gate(0.9, 1.0, 100.0) == 1.0
    assert apply_presence_gate(0.0, 1.0, 100.0) == 0.0


# ----- _load_bird_mask on a SYNTHETIC asset (SC-C2), no TF -----------------


def _write_ebird_asset(model_dir, rows: list[str], header: str | None = None) -> None:
    assets = model_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    lines = ([header] if header else []) + rows
    (assets / "perch_v2_ebird_classes.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_load_bird_mask_mixed_rows(tmp_path) -> None:
    _write_ebird_asset(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b", "c"])
    mask = adapter._load_bird_mask()
    assert mask is not None
    assert list(mask) == [True, False, True]


def test_load_bird_mask_drops_sentinel_header(tmp_path) -> None:
    _write_ebird_asset(tmp_path, ["amerob", "no_ebird_code"], header="ebird2021")
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b"])
    mask = adapter._load_bird_mask()
    assert list(mask) == [True, False]  # header dropped, 2 class rows


def test_load_bird_mask_missing_file_is_none(tmp_path, caplog) -> None:
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b"])
    import logging

    with caplog.at_level(logging.WARNING):
        assert adapter._load_bird_mask() is None
    assert "presence gate disabled" in caplog.text


def test_load_bird_mask_is_cached(tmp_path) -> None:
    _write_ebird_asset(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b", "c"])
    first = adapter._load_bird_mask()
    # Delete the asset; a cached load must still return the same mask.
    (tmp_path / "assets" / "perch_v2_ebird_classes.csv").unlink()
    assert adapter._load_bird_mask() is first


# ----- classify() gate integration (TF-free via the fake model) ------------


def _gate_adapter(tmp_path, ebird_rows: list[str]) -> Perch2Adapter:
    _write_ebird_asset(tmp_path, ebird_rows)
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b", "c"])
    adapter._model = _FakeModel(_FakeSignature())  # logits [0.1, 0.9, 0.5]
    return adapter


def test_classify_k0_returns_raw_logits(tmp_path, monkeypatch) -> None:
    """k=0 (default) is the LITERAL raw-logit path — not the gate at k=0."""
    monkeypatch.delenv("FAUN_PRESENCE_GATE_K", raising=False)
    get_settings.cache_clear()
    adapter = _gate_adapter(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    # The fake's raw logits, verbatim (top is class 1 = 0.9).
    assert preds[0].probability == pytest.approx(0.9)
    assert {round(p.probability, 4) for p in preds} == {0.9, 0.5, 0.1}


def test_classify_gate_on_rescales_to_unit_interval(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_PRESENCE_GATE_K", "2.0")
    get_settings.cache_clear()
    adapter = _gate_adapter(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    # Output now changes UNDER the flag: gated softmax probs, all within [0,1].
    assert all(0.0 <= p.probability <= 1.0 for p in preds)
    assert preds[0].probability != pytest.approx(0.9)  # no longer the raw logit
    # Ranking is unchanged (uniform per-segment multiplier): class 1 still top.
    assert preds[0].species == "b"
    # Exact value: clamp01(softmax(scores)[1] * (1 + p_bird*2)).
    scores = np.array([0.1, 0.9, 0.5])
    probs = _softmax(scores)
    p_bird = float(probs[0] + probs[2])
    assert preds[0].probability == pytest.approx(
        min(1.0, probs[1] * (1 + p_bird * 2.0))
    )


def test_classify_gate_length_mismatch_disables_fail_open(
    tmp_path, monkeypatch, caplog
) -> None:
    """A bird mask not 1:1 with the logits disables the gate (raw logits kept)."""
    monkeypatch.setenv("FAUN_PRESENCE_GATE_K", "2.0")
    get_settings.cache_clear()
    # 4 rows but the fake has 3 classes -> length mismatch.
    adapter = _gate_adapter(tmp_path, ["a1", "no_ebird_code", "a2", "a3"])
    import logging

    with caplog.at_level(logging.WARNING):
        preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].probability == pytest.approx(0.9)  # raw logit, gate skipped
    assert "!= logit count" in caplog.text


def test_classify_gate_missing_mask_disables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_PRESENCE_GATE_K", "2.0")
    get_settings.cache_clear()
    # No eBird asset written -> mask None -> gate no-op even with k>0.
    adapter = Perch2Adapter(model_path=str(tmp_path), labels=["a", "b", "c"])
    adapter._model = _FakeModel(_FakeSignature())
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].probability == pytest.approx(0.9)


# ----- classify() calibration integration (FR-006-serve) -------------------


def test_classify_no_calibrator_leaves_prob_calibrated_none(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("PERCH_V2_CALIBRATOR_PATH", raising=False)
    get_settings.cache_clear()
    adapter = _gate_adapter(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert all(p.prob_calibrated is None for p in preds)


def test_classify_with_calibrator_sets_unit_interval_prob(
    tmp_path, monkeypatch
) -> None:
    from faun.retraining import TemperatureCalibrator, save_probe

    cal_path = tmp_path / "cal.pkl"
    save_probe(TemperatureCalibrator(temperature=2.0), cal_path)
    monkeypatch.setenv("PERCH_V2_CALIBRATOR_PATH", str(cal_path))
    monkeypatch.delenv("FAUN_PRESENCE_GATE_K", raising=False)  # gate off
    get_settings.cache_clear()

    adapter = _gate_adapter(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)

    # Raw probability untouched (still the logit); calibrated is softmax(logits/T).
    assert preds[0].probability == pytest.approx(0.9)
    scores = np.array([0.1, 0.9, 0.5])
    expected = _softmax(scores / 2.0)
    assert preds[0].prob_calibrated == pytest.approx(float(expected[1]))
    assert all(0.0 <= p.prob_calibrated <= 1.0 for p in preds)


def test_classify_bad_calibrator_path_is_fail_open(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PERCH_V2_CALIBRATOR_PATH", str(tmp_path / "nope.pkl"))
    get_settings.cache_clear()
    adapter = _gate_adapter(tmp_path, ["amerob", "no_ebird_code", "comrav"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert all(p.prob_calibrated is None for p in preds)  # no crash, None
