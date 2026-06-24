"""TF-free tests for PerchProbeAdapter (Perch 2 embeddings + trained probe).

The Perch 2 embedder is injected as a fake (so TensorFlow / kagglehub are never
imported); the probe is a tiny stand-in exposing the sklearn surface
(``predict_proba`` / ``classes_``). These pin: probe ranking + class naming,
the no-probe embedding_only path, explicit-labels precedence, and that the
PERCH_V2_PROBE_PATH env is honoured via faun.settings.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np

from faun.classification import Prediction
from faun.classification.perch_probe import EMBEDDING_ONLY, PerchProbeAdapter


class _FakeEmbedder:
    """Stand-in for Perch2Adapter: embed() returns a fixed 1536-d vector."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, waveform, sr):
        self.calls += 1
        return np.arange(1536, dtype=np.float32)


class _FakeProbe:
    """sklearn-ish probe: predict_proba + classes_ (species names)."""

    classes_ = np.array(["Fringilla coelebs", "Parus major", "Turdus merula"])

    def predict_proba(self, x):
        # class index 2 ("Turdus merula") wins.
        n = np.asarray(x).shape[0]
        return np.tile(np.array([0.1, 0.2, 0.7]), (n, 1))


def _adapter(**kw) -> PerchProbeAdapter:
    adapter = PerchProbeAdapter(**kw)
    adapter._embedder = _FakeEmbedder()  # inject so TF is never imported
    return adapter


def test_classify_ranks_by_probe_and_names_from_classes():
    adapter = _adapter(probe=_FakeProbe())
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert all(isinstance(p, Prediction) for p in preds)
    assert preds[0].species == "Turdus merula"  # highest proba
    probs = [p.probability for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_no_probe_returns_embedding_only_and_stashes_embedding():
    adapter = _adapter()  # no probe, no probe_path
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds == [Prediction(EMBEDDING_ONLY, 0.0)]
    assert adapter.last_embedding is not None
    assert adapter.last_embedding.shape == (1536,)


def test_embed_delegates_to_embedder():
    adapter = _adapter()
    emb = adapter.embed(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert emb.shape == (1536,)
    assert adapter._embedder.calls == 1


def test_explicit_labels_take_precedence_over_classes():
    adapter = _adapter(probe=_FakeProbe(), labels=["L0", "L1", "L2"])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    # index 2 still wins, but the name comes from the explicit labels arg.
    assert preds[0].species == "L2"


def test_top_k_caps_predictions():
    adapter = _adapter(probe=_FakeProbe(), top_k=2)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert len(preds) == 2


def test_probe_path_from_settings_env(monkeypatch, tmp_path):
    """PERCH_V2_PROBE_PATH (via faun.settings) is picked up + the pickle loads."""
    probe_file = tmp_path / "probe.pkl"
    with open(probe_file, "wb") as fh:
        pickle.dump(_FakeProbe(), fh)
    monkeypatch.setenv("PERCH_V2_PROBE_PATH", str(probe_file))
    from faun.settings import get_settings

    get_settings.cache_clear()
    adapter = PerchProbeAdapter()
    adapter._embedder = _FakeEmbedder()
    assert adapter.probe_path == str(probe_file)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].species == "Turdus merula"  # loaded probe ranks


def test_module_import_is_tf_free():
    """Importing the adapter must not pull TensorFlow or kagglehub."""
    sys.modules.pop("faun.classification.perch_probe", None)
    import faun.classification.perch_probe  # noqa: F401

    assert "tensorflow" not in sys.modules
    assert "kagglehub" not in sys.modules


# ---------------------------------------------------------------------------
# FR-007: the prototype probe (with a negative class) is a drop-in for the
# UNCHANGED PerchProbeAdapter — round-trips through save_probe/load_probe and
# classifies via the existing predict_proba/classes_ path (no TF).
# ---------------------------------------------------------------------------


def _clustered(seed: int, *, dim: int = 1536, per: int = 12):
    """Three bird clusters + a separated negative cluster; returns centers too."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((4, dim)) * 6.0

    def cloud(c, n):
        return c + 0.05 * rng.standard_normal((n, dim))

    X = np.vstack([cloud(centers[i], per) for i in range(3)])
    y = np.array(sum(([f"sp{i}"] * per for i in range(3)), []))
    negatives = cloud(centers[3], per)
    return X, y, negatives, centers


class _CentroidEmbedder:
    """Embedder stand-in that always returns a fixed 1536-d vector (a centroid)."""

    def __init__(self, vec) -> None:
        self.vec = np.asarray(vec, dtype=np.float32)

    def embed(self, waveform, sr):
        return self.vec


def test_prototype_probe_round_trip_classifies_via_adapter(tmp_path):
    """SC-D1: save -> load -> PerchProbeAdapter(probe=loaded) classifies (no TF)."""
    from faun.retraining import (
        NEGATIVE_CLASS,
        load_probe,
        save_probe,
        train_prototype_probe,
    )

    X, y, negatives, centers = _clustered(seed=11)
    probe = train_prototype_probe(X, y, negatives=negatives)

    out = tmp_path / "prototype.pkl"
    save_probe(probe, out)
    loaded = load_probe(out)
    assert np.array_equal(loaded.classes_, probe.classes_)

    # A query AT the sp0 bird centroid -> top prediction is sp0.
    adapter = PerchProbeAdapter(probe=loaded, top_k=5)
    adapter._embedder = _CentroidEmbedder(centers[0])
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert all(isinstance(p, Prediction) for p in preds)
    assert preds[0].species == "sp0"
    probs = [p.probability for p in preds]
    assert probs == sorted(probs, reverse=True)

    # A query AT the negative centroid -> the adapter surfaces NEGATIVE_CLASS.
    adapter_neg = PerchProbeAdapter(probe=loaded, top_k=5)
    adapter_neg._embedder = _CentroidEmbedder(centers[3])
    preds_neg = adapter_neg.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds_neg[0].species == NEGATIVE_CLASS
