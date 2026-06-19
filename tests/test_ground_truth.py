"""Single-home contract for ground truth + training-excluded sources.

``faun.detections`` is the ONE definition of both the ``is_ground_truth``
predicate and the ``TRAINING_EXCLUDED_SOURCES`` set; ``faun.retraining`` and
``faun.labeling`` import them instead of mirroring. This module proves that
contract with a source × status matrix and identity assertions — TF/torch-free.
"""

from __future__ import annotations

import faun.detections as detections
import faun.labeling as labeling
import faun.retraining as retraining
from faun.detections import (
    SOURCE_BIRDNET,
    SOURCE_EXPERT,
    SOURCE_PERCH,
    SOURCE_RANGER,
    STATUS_CONFIRMED,
    STATUS_CORRECTED,
    STATUS_PSEUDO,
    TRAINING_EXCLUDED_SOURCES,
    Detection,
    Label,
    is_ground_truth,
)
from faun.labeling import training_candidates
from faun.segmentation import Segment

# Sources: two model arms, expert, operator, plus empty / junk for the
# not-a-string / unknown-prefix branches.
_SOURCES = [SOURCE_PERCH, SOURCE_BIRDNET, SOURCE_EXPERT, SOURCE_RANGER, "", "junk"]
# Statuses: pseudo + the two ground-truth statuses + empty / junk.
_STATUSES = [STATUS_PSEUDO, STATUS_CONFIRMED, STATUS_CORRECTED, "", "junk"]


def _expected(source: str, status: str) -> bool:
    """The documented contract, computed independently of the module under test."""
    return source.startswith(("expert:", "operator:")) and status in (
        STATUS_CONFIRMED,
        STATUS_CORRECTED,
    )


# ---------------------------------------------------------------------------
# Source × status matrix — Label object AND dict agree with the contract
# ---------------------------------------------------------------------------


def test_ground_truth_matrix_object_and_dict_agree():
    for source in _SOURCES:
        for status in _STATUSES:
            expected = _expected(source, status)
            label = Label.now("Turdus merula", 0.5, source, status)
            mapping = {"source": source, "status": status}

            obj_result = is_ground_truth(label)
            dict_result = is_ground_truth(mapping)

            assert obj_result is expected, (source, status, "object")
            assert dict_result is expected, (source, status, "dict")
            # Object and dict must give identical answers — same single home.
            assert obj_result == dict_result, (source, status)


def test_ground_truth_handles_missing_and_non_string_fields():
    # Missing keys / attributes -> not a string -> False.
    assert is_ground_truth({}) is False
    assert is_ground_truth({"source": SOURCE_EXPERT}) is False
    assert is_ground_truth({"status": STATUS_CONFIRMED}) is False
    # Non-string field values -> False (no crash).
    assert is_ground_truth({"source": None, "status": STATUS_CONFIRMED}) is False
    assert is_ground_truth({"source": SOURCE_EXPERT, "status": 1}) is False


# ---------------------------------------------------------------------------
# Single-home identity assertions
# ---------------------------------------------------------------------------


def test_retraining_reuses_detections_predicate():
    # Identity: retraining must not redefine the predicate, only import it.
    assert retraining.is_ground_truth is detections.is_ground_truth


def test_labeling_reuses_detections_excluded_set():
    # Identity / equality: labeling must not mirror the excluded-source set.
    assert labeling.TRAINING_EXCLUDED_SOURCES is detections.TRAINING_EXCLUDED_SOURCES
    assert labeling.TRAINING_EXCLUDED_SOURCES == TRAINING_EXCLUDED_SOURCES
    assert SOURCE_BIRDNET in TRAINING_EXCLUDED_SOURCES


def test_training_candidates_drops_birdnet_via_shared_set():
    det = Detection.new(
        trap_id="A1",
        source_file="rec.wav",
        segment=Segment(start_s=0.0, duration_s=1.0),
        labels=[
            Label.now("Turdus merula", 0.9, SOURCE_PERCH, STATUS_PSEUDO),
            Label.now("Turdus merula", 0.8, SOURCE_BIRDNET, STATUS_PSEUDO),
        ],
    )
    candidates = training_candidates([det])

    survived = [lbl.source for cand in candidates for lbl in cand.labels]
    assert SOURCE_PERCH in survived
    assert SOURCE_BIRDNET not in survived
