"""TF-free tests for two-model routing (Wave 2). Mirrors tests/test_perch_v2.py's
fake-SavedModel injection; never imports TensorFlow."""

from __future__ import annotations

import numpy as np
import pytest

from faun.classification import Prediction, RoutingClassifier
from faun.classification.perch_v2 import (
    PERCH_V2_DIM,
    Perch2Adapter,
    RoutingResult,
    _softmax,
    bird_presence_mass,
)
from faun.detections import STATUS_PSEUDO, STATUS_REJECTED
from faun.settings import get_settings


class _FakeSignature:
    """Perch 2 serving stand-in: 3-class logits, class 1 wins."""

    def __init__(self, dim: int = PERCH_V2_DIM, n_classes: int = 3) -> None:
        self.dim = dim
        self.n_classes = n_classes
        self.last_inputs: np.ndarray | None = None

    def __call__(self, inputs):
        self.last_inputs = np.asarray(inputs)
        n = self.last_inputs.shape[0]
        logits = np.tile(np.array([0.1, 0.9, 0.5], dtype=np.float32), (n, 1))
        return {
            "embedding": np.zeros((n, self.dim), dtype=np.float32),
            "label": logits[:, : self.n_classes],
        }


class _FakeModel:
    def __init__(self, sig: _FakeSignature) -> None:
        self.signatures = {"serving_default": sig}


def _adapter(monkeypatch, mask):
    a = Perch2Adapter(model_path="/fake", labels=["a", "b", "c"])
    a._model = _FakeModel(_FakeSignature())
    monkeypatch.setattr(a, "_load_bird_mask", lambda: mask)
    return a


# ---------------------------------------------------------------------------
# bird_presence_mass math (unit, no adapter)
# ---------------------------------------------------------------------------


def test_bird_presence_mass_all_bird():
    assert bird_presence_mass(
        np.array([1.0, 2.0, 3.0]), np.array([True, True, True])
    ) == pytest.approx(1.0)


def test_bird_presence_mass_all_noise():
    assert bird_presence_mass(
        np.array([1.0, 2.0, 3.0]), np.array([False, False, False])
    ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Perch2Adapter.classify_with_routing
# ---------------------------------------------------------------------------


def test_classify_with_routing_matches_classify(monkeypatch):
    """Refactor guard: routing predictions == classify() for the same scores."""
    a = _adapter(monkeypatch, np.array([True, True, False]))
    result = a.classify_with_routing(np.zeros(32_000, np.float32), 32_000)
    assert isinstance(result, RoutingResult)
    assert result.predictions == a.classify(np.zeros(32_000, np.float32), 32_000)
    probs = _softmax(np.array([0.1, 0.9, 0.5]))
    assert result.p_bird == pytest.approx(float(probs[0] + probs[1]))
    assert result.non_bird_top == ("c", pytest.approx(0.5))  # only non-bird class


def test_classify_with_routing_mask_none_fail_open(monkeypatch):
    a = _adapter(monkeypatch, None)
    result = a.classify_with_routing(np.zeros(32_000, np.float32), 32_000)
    assert result.p_bird is None and result.non_bird_top is None


def test_classify_with_routing_mask_length_mismatch_fail_open(monkeypatch):
    """A wrong-length mask disables routing (fail-open), like the label guard."""
    a = _adapter(monkeypatch, np.array([True, False]))  # 2 != 3 logits
    result = a.classify_with_routing(np.zeros(32_000, np.float32), 32_000)
    assert result.p_bird is None and result.non_bird_top is None


# ---------------------------------------------------------------------------
# RoutingClassifier decision logic
# ---------------------------------------------------------------------------


def test_routing_wrapper_rejects_low_p_bird():
    class Inner:
        def classify_with_routing(self, s, sr):
            return RoutingResult([Prediction("x", 1.0)], p_bird=0.1, non_bird_top=None)

    preds, status = RoutingClassifier(Inner(), tau_bird=0.5).classify_with_status(
        None, 0
    )
    assert status == STATUS_REJECTED and preds[0].species == "x"


def test_routing_wrapper_keeps_high_p_bird():
    class Inner:
        def classify_with_routing(self, s, sr):
            return RoutingResult([Prediction("x", 1.0)], p_bird=0.9, non_bird_top=None)

    assert (
        RoutingClassifier(Inner(), tau_bird=0.5).classify_with_status(None, 0)[1]
        == STATUS_PSEUDO
    )


def test_routing_wrapper_p_bird_none_is_pseudo():
    class Inner:
        def classify_with_routing(self, s, sr):
            return RoutingResult([Prediction("x", 1.0)], p_bird=None, non_bird_top=None)

    assert (
        RoutingClassifier(Inner(), tau_bird=0.5).classify_with_status(None, 0)[1]
        == STATUS_PSEUDO
    )


def test_routing_wrapper_passthrough_non_router():
    class Inner:
        def classify(self, s, sr):
            return [Prediction("x", 1.0)]

    preds, status = RoutingClassifier(Inner(), tau_bird=0.5).classify_with_status(
        None, 0
    )
    assert status == STATUS_PSEUDO and preds[0].species == "x"


def test_routing_wrapper_classify_protocol():
    class Inner:
        def classify_with_routing(self, s, sr):
            return RoutingResult([Prediction("x", 1.0)], p_bird=0.1, non_bird_top=None)

    preds = RoutingClassifier(Inner(), tau_bird=0.5).classify(None, 0)
    assert [p.species for p in preds] == ["x"]  # just predictions, no status


# ---------------------------------------------------------------------------
# MaskedClassifier + routing: status survives the mask
# ---------------------------------------------------------------------------


def test_masked_preserves_routing_status():
    from faun.classification import MaskedClassifier

    class Inner:
        def classify_with_status(self, s, sr):
            return [Prediction("Turdus merula", 1.0)], STATUS_REJECTED

    m = MaskedClassifier(
        Inner(),
        ["Turdus merula"],
        vocab_provider=lambda: ["Turdus merula", "Parus major"],
    )
    preds, status = m.classify_with_status(None, 0)
    assert status == STATUS_REJECTED and preds[0].species == "Turdus merula"


# ---------------------------------------------------------------------------
# api._build_classifier wiring
# ---------------------------------------------------------------------------


def test_build_classifier_routing_on_off(monkeypatch):
    from faun import api

    monkeypatch.setenv("FAUN_CLASSIFIER", "stub")
    monkeypatch.setenv("FAUN_ROUTING_ENABLED", "1")
    get_settings.cache_clear()
    assert hasattr(api._build_classifier(), "classify_with_status")
    monkeypatch.delenv("FAUN_ROUTING_ENABLED")
    get_settings.cache_clear()
    assert not hasattr(api._build_classifier(), "classify_with_status")
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# _build_labels end-to-end: a rejecting classifier stamps STATUS_REJECTED
# ---------------------------------------------------------------------------


def test_build_labels_emits_rejected(tmp_path):
    """FR-R2 end-to-end: an injected rejecting classifier makes Label.status
    == rejected in detections.jsonl (drives the REAL run_pipeline chain)."""
    from faun.api import run_pipeline
    from faun.detections import read_detections
    from tests.pipeline.golden_util import make_trap_dir

    make_trap_dir(tmp_path / "data")

    class _Rejecting:
        def classify_with_status(self, segment, sr):
            return [Prediction("Turdus merula", 13.3)], STATUS_REJECTED

    job_dir = tmp_path / "job"
    run_pipeline(job_dir, str(tmp_path / "data"), classifier=_Rejecting())

    dets = read_detections(job_dir / "detections.jsonl")
    assert dets, "burst must still produce a detection"
    labels = [lbl for det in dets for lbl in det.labels]
    assert labels and all(lbl.status == STATUS_REJECTED for lbl in labels)
