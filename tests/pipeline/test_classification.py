"""Tests for faun.classification adapters (BirdNET / YAMNet / Perch).

All heavy dependencies (birdnetlib, tensorflow, tensorflow_hub) are mocked —
either by patching the adapter's internal ``_load`` / ``_load_models`` hooks or
by injecting fake modules into ``sys.modules`` (see tests/test_classifier.py for
the in-tree mocking pattern). These tests never import a real ML model.

Coverage:
  * module imports succeed without any heavy lib present;
  * lazy package re-exports (PEP 562) work and don't pull heavy modules;
  * each adapter's classify() returns list[Prediction] with descending
    probability on a synthetic signal + mocked model;
  * resampling is invoked with the right (in_sr, target_sr);
  * RuntimeError when the heavy lib is missing;
  * StubAdapter satisfies the runtime_checkable Protocol.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faun.classification import Prediction, SpeciesClassifier, StubAdapter


# ---------------------------------------------------------------------------
# Module imports / lazy package exports
# ---------------------------------------------------------------------------


class TestImportsAreLight:
    def test_birdnet_module_imports_without_birdnetlib(self) -> None:
        """birdnet.py imports cleanly even though birdnetlib is absent."""
        assert "birdnetlib" not in sys.modules
        import faun.classification.birdnet as mod

        assert hasattr(mod, "BirdNETAdapter")
        assert "birdnetlib" not in sys.modules

    def test_yamnet_module_imports_without_tensorflow(self) -> None:
        import faun.classification.yamnet as mod

        assert hasattr(mod, "YAMNetAdapter")
        assert "tensorflow" not in sys.modules

    def test_perch_module_imports_without_tensorflow(self) -> None:
        import faun.classification.perch as mod

        assert hasattr(mod, "PerchAdapter")
        assert "tensorflow_hub" not in sys.modules

    def test_lazy_reexports_resolve(self) -> None:
        from faun.classification import (  # noqa: F401
            BirdNETAdapter,
            PerchAdapter,
            YAMNetAdapter,
        )

        assert BirdNETAdapter.__name__ == "BirdNETAdapter"
        assert YAMNetAdapter.__name__ == "YAMNetAdapter"
        assert PerchAdapter.__name__ == "PerchAdapter"

    def test_unknown_package_attribute_raises(self) -> None:
        import faun.classification as pkg

        with pytest.raises(AttributeError):
            _ = pkg.NopeAdapter


# ---------------------------------------------------------------------------
# Stub / Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_stub_is_species_classifier(self) -> None:
        assert isinstance(StubAdapter(), SpeciesClassifier)

    def test_adapters_are_species_classifiers(self) -> None:
        from faun.classification import (
            BirdNETAdapter,
            PerchAdapter,
            YAMNetAdapter,
        )

        assert isinstance(BirdNETAdapter(), SpeciesClassifier)
        assert isinstance(YAMNetAdapter(), SpeciesClassifier)
        assert isinstance(PerchAdapter(), SpeciesClassifier)

    def test_object_without_classify_is_not_classifier(self) -> None:
        assert not isinstance(object(), SpeciesClassifier)


# ---------------------------------------------------------------------------
# BirdNETAdapter
# ---------------------------------------------------------------------------


def _make_fake_birdnetlib(detections: list[dict]):
    """Build fake ``birdnetlib`` + ``birdnetlib.analyzer`` modules.

    Returns ``(modules_dict, recording_cls)`` where ``modules_dict`` is suitable
    for ``patch.dict(sys.modules, ...)``.
    """
    analyzer_mod = types.ModuleType("birdnetlib.analyzer")
    analyzer_mod.Analyzer = MagicMock(name="Analyzer")

    root_mod = types.ModuleType("birdnetlib")
    root_mod.analyzer = analyzer_mod

    class FakeRecording:
        last_kwargs: dict = {}

        def __init__(self, analyzer, path, **kwargs):
            FakeRecording.last_kwargs = kwargs
            self.detections = []

        def analyze(self):
            self.detections = detections

    root_mod.Recording = FakeRecording
    return (
        {"birdnetlib": root_mod, "birdnetlib.analyzer": analyzer_mod},
        FakeRecording,
    )


class TestBirdNET:
    def test_classify_returns_ranked_predictions(self) -> None:
        from faun.classification import BirdNETAdapter

        dets = [
            {"scientific_name": "Turdus merula", "confidence": 0.91},
            {"scientific_name": "Parus major", "confidence": 0.40},
        ]
        modules, _ = _make_fake_birdnetlib(dets)
        adapter = BirdNETAdapter()
        waveform = np.random.randn(48_000).astype(np.float32)

        with patch.dict(sys.modules, modules):
            preds = adapter.classify(waveform, 48_000)

        assert isinstance(preds, list)
        assert all(isinstance(p, Prediction) for p in preds)
        assert [p.species for p in preds] == ["Turdus merula", "Parus major"]
        assert preds[0].probability >= preds[1].probability

    def test_classify_resamples_to_48k(self) -> None:
        from faun.classification import BirdNETAdapter

        modules, _ = _make_fake_birdnetlib(
            [{"scientific_name": "X", "confidence": 0.5}]
        )
        adapter = BirdNETAdapter()
        waveform = np.random.randn(16_000).astype(np.float32)

        with (
            patch.dict(sys.modules, modules),
            patch("faun.classification.birdnet.soxr") as mock_soxr,
        ):
            mock_soxr.resample.return_value = np.zeros(48_000, dtype=np.float32)
            adapter.classify(waveform, 16_000)

        mock_soxr.resample.assert_called_once()
        args = mock_soxr.resample.call_args[0]
        assert args[1] == 16_000
        assert args[2] == 48_000

    def test_classify_passes_lat_lon_date(self) -> None:
        from faun.classification import BirdNETAdapter

        modules, recording_cls = _make_fake_birdnetlib(
            [{"scientific_name": "X", "confidence": 0.5}]
        )
        adapter = BirdNETAdapter(lat=58.0, lon=44.0, date="d", min_conf=0.25)
        waveform = np.random.randn(48_000).astype(np.float32)

        with patch.dict(sys.modules, modules):
            adapter.classify(waveform, 48_000)

        assert recording_cls.last_kwargs["lat"] == 58.0
        assert recording_cls.last_kwargs["lon"] == 44.0
        assert recording_cls.last_kwargs["date"] == "d"
        assert recording_cls.last_kwargs["min_conf"] == 0.25

    def test_classify_raises_when_birdnetlib_missing(self) -> None:
        from faun.classification import BirdNETAdapter

        adapter = BirdNETAdapter()
        waveform = np.random.randn(48_000).astype(np.float32)

        # Force the import inside _load to fail.
        with patch.dict(sys.modules, {"birdnetlib.analyzer": None}):
            with pytest.raises(RuntimeError, match="birdnetlib"):
                adapter.classify(waveform, 48_000)


# ---------------------------------------------------------------------------
# YAMNetAdapter
# ---------------------------------------------------------------------------


def _make_yamnet_model_mock(emb_shape: tuple[int, int] = (5, 1024)) -> MagicMock:
    model = MagicMock(name="yamnet")
    scores = MagicMock()
    scores.numpy.return_value = np.random.rand(emb_shape[0], 521).astype(np.float32)
    embeddings = MagicMock()
    embeddings.numpy.return_value = np.random.randn(*emb_shape).astype(np.float32)
    spec = MagicMock()
    model.return_value = (scores, embeddings, spec)
    return model


class TestYAMNet:
    def test_classify_embedding_only_without_probe(self) -> None:
        from faun.classification import YAMNetAdapter

        adapter = YAMNetAdapter()
        model = _make_yamnet_model_mock()
        waveform = np.random.randn(16_000).astype(np.float32)

        with patch("faun.ml.yamnet._load_models", return_value=(model, None)):
            preds = adapter.classify(waveform, 16_000)

        assert preds == [Prediction("embedding_only", 0.0)]
        # Embedding stashed for downstream few-shot use.
        assert adapter.last_embedding is not None
        assert adapter.last_embedding.shape == (1024,)

    def test_classify_with_probe_returns_ranked(self) -> None:
        from faun.classification import YAMNetAdapter

        probe = MagicMock(name="probe")
        probe.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        adapter = YAMNetAdapter(probe=probe, labels=["a", "b", "c"])
        model = _make_yamnet_model_mock()
        waveform = np.random.randn(16_000).astype(np.float32)

        with patch("faun.ml.yamnet._load_models", return_value=(model, None)):
            preds = adapter.classify(waveform, 16_000)

        assert [p.species for p in preds] == ["b", "c", "a"]
        probs = [p.probability for p in preds]
        assert probs == sorted(probs, reverse=True)

    def test_embed_resamples_to_16k(self) -> None:
        from faun.classification import YAMNetAdapter

        adapter = YAMNetAdapter()
        model = _make_yamnet_model_mock()
        waveform = np.random.randn(48_000).astype(np.float32)

        with (
            patch("faun.ml.yamnet._load_models", return_value=(model, None)),
            patch("faun.classification.yamnet.soxr") as mock_soxr,
        ):
            mock_soxr.resample.return_value = np.zeros(16_000, dtype=np.float32)
            adapter.embed(waveform, 48_000)

        mock_soxr.resample.assert_called_once()
        args = mock_soxr.resample.call_args[0]
        assert args[1] == 48_000
        assert args[2] == 16_000

    def test_embed_returns_1024_dim(self) -> None:
        from faun.classification import YAMNetAdapter

        adapter = YAMNetAdapter()
        model = _make_yamnet_model_mock(emb_shape=(3, 1024))
        waveform = np.random.randn(16_000).astype(np.float32)

        with patch("faun.ml.yamnet._load_models", return_value=(model, None)):
            emb = adapter.embed(waveform, 16_000)

        assert emb.shape == (1024,)


# ---------------------------------------------------------------------------
# PerchAdapter
# ---------------------------------------------------------------------------


def _make_perch_model_mock(n_species: int = 10, emb_dim: int = 1280) -> MagicMock:
    model = MagicMock(name="perch")
    logits = np.random.randn(1, n_species).astype(np.float32)
    embedding = np.random.randn(1, emb_dim).astype(np.float32)
    model.infer_tf.return_value = (logits, embedding)
    return model


class TestPerch:
    def test_default_source_is_tfhub(self) -> None:
        from faun.classification import PerchAdapter
        from faun.classification.perch import DEFAULT_TFHUB_URL

        assert PerchAdapter().model_path == DEFAULT_TFHUB_URL

    def test_env_overrides_source(self, monkeypatch) -> None:
        from faun.classification import PerchAdapter

        monkeypatch.setenv("PERCH_MODEL_PATH", "/models/perch")
        assert PerchAdapter().model_path == "/models/perch"

    def test_classify_returns_ranked_predictions(self) -> None:
        from faun.classification import PerchAdapter

        adapter = PerchAdapter()
        logits = np.array([[0.1, 0.9, 0.5, 0.2]], dtype=np.float32)
        emb = np.random.randn(1, 1280).astype(np.float32)
        model = MagicMock()
        model.infer_tf.return_value = (logits, emb)
        waveform = np.random.randn(32_000).astype(np.float32)

        with patch.object(adapter, "_load", return_value=model):
            preds = adapter.classify(waveform, 32_000)

        assert all(isinstance(p, Prediction) for p in preds)
        probs = [p.probability for p in preds]
        assert probs == sorted(probs, reverse=True)
        assert preds[0].species == "species_1"  # argmax of logits

    def test_classify_resamples_and_pads_to_5s(self) -> None:
        from faun.classification import PerchAdapter
        from faun.classification.perch import PERCH_WINDOW_SAMPLES

        adapter = PerchAdapter()
        model = _make_perch_model_mock()
        # 1 s at 16 kHz -> resampled to 32 kHz then padded to 5 s.
        waveform = np.random.randn(16_000).astype(np.float32)

        with (
            patch.object(adapter, "_load", return_value=model),
            patch("faun.classification.perch.soxr") as mock_soxr,
        ):
            mock_soxr.resample.return_value = np.zeros(32_000, dtype=np.float32)
            adapter.classify(waveform, 16_000)

        mock_soxr.resample.assert_called_once()
        args = mock_soxr.resample.call_args[0]
        assert args[1] == 16_000
        assert args[2] == 32_000
        fed = model.infer_tf.call_args[0][0]
        assert fed.shape == (1, PERCH_WINDOW_SAMPLES)

    def test_classify_crops_long_signal_to_5s(self) -> None:
        from faun.classification import PerchAdapter
        from faun.classification.perch import PERCH_WINDOW_SAMPLES

        adapter = PerchAdapter()
        model = _make_perch_model_mock()
        # 10 s at 32 kHz -> cropped to 5 s, no resample.
        waveform = np.random.randn(32_000 * 10).astype(np.float32)

        with patch.object(adapter, "_load", return_value=model):
            adapter.classify(waveform, 32_000)

        fed = model.infer_tf.call_args[0][0]
        assert fed.shape == (1, PERCH_WINDOW_SAMPLES)

    def test_embed_returns_embedding(self) -> None:
        from faun.classification import PerchAdapter

        adapter = PerchAdapter()
        model = _make_perch_model_mock(emb_dim=1280)
        waveform = np.random.randn(32_000 * 5).astype(np.float32)

        with patch.object(adapter, "_load", return_value=model):
            emb = adapter.embed(waveform, 32_000)

        assert emb.shape == (1280,)

    def test_classify_raises_when_tf_missing(self) -> None:
        from faun.classification import PerchAdapter

        adapter = PerchAdapter()
        waveform = np.random.randn(32_000).astype(np.float32)

        with patch.dict(sys.modules, {"tensorflow_hub": None}):
            with pytest.raises(RuntimeError, match="tensorflow_hub"):
                adapter.classify(waveform, 32_000)
