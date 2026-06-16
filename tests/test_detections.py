"""Tests for faun.detections — the central detection abstraction.

Covers JSONL round-trip fidelity, the one-object-per-line contract, atomic
overwrite (no leftover .tmp), the ground-truth predicate gate, and
Prediction -> Label lifting.
"""

from __future__ import annotations

from pathlib import Path

from faun.classification import Prediction
from faun.detections import (
    SOURCE_BIRDNET,
    SOURCE_EXPERT,
    SOURCE_PERCH,
    SOURCE_RANGER,
    SOURCE_YAMNET_PROBE,
    STATUS_CONFIRMED,
    STATUS_CORRECTED,
    STATUS_PSEUDO,
    Detection,
    Label,
    is_ground_truth,
    read_detections,
    write_detections,
)
from faun.segmentation import Segment


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _detection(det_id: str, trap: str = "A1") -> Detection:
    seg = Segment(start_s=12.5, duration_s=3.0)
    labels = [
        Label.now("Parus major", 0.91, SOURCE_PERCH, STATUS_PSEUDO),
        Label.now("Erithacus rubecula", None, SOURCE_EXPERT, STATUS_CONFIRMED),
    ]
    return Detection(
        detection_id=det_id,
        trap_id=trap,
        source_file="REC_20260610_213000.wav",
        segment=seg,
        segment_path=f"segments/{det_id}.wav",
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    dets = [_detection("a1b2"), _detection("c3d4", trap="A2")]
    path = tmp_path / "detections.jsonl"

    write_detections(path, dets)
    loaded = read_detections(path)

    assert len(loaded) == len(dets)
    for original, restored in zip(dets, loaded):
        assert restored.detection_id == original.detection_id
        assert restored.trap_id == original.trap_id
        assert restored.source_file == original.source_file
        assert restored.segment_path == original.segment_path
        assert restored.segment.start_s == original.segment.start_s
        assert restored.segment.duration_s == original.segment.duration_s
        assert restored.segment.end_s == original.segment.end_s
        assert len(restored.labels) == len(original.labels)
        for lab_o, lab_r in zip(original.labels, restored.labels):
            assert lab_r.species == lab_o.species
            assert lab_r.probability == lab_o.probability
            assert lab_r.source == lab_o.source
            assert lab_r.status == lab_o.status
            assert lab_r.ts == lab_o.ts


def test_round_trip_segment_dict_keys(tmp_path: Path) -> None:
    det = _detection("ff00")
    payload = det.to_dict()
    assert set(payload["segment"]) == {"start_s", "duration_s"}
    assert Detection.from_dict(payload).segment == det.segment


# ---------------------------------------------------------------------------
# JSONL contract
# ---------------------------------------------------------------------------


def test_one_json_object_per_line(tmp_path: Path) -> None:
    dets = [_detection("a"), _detection("b"), _detection("c")]
    path = tmp_path / "out.jsonl"

    write_detections(path, dets)

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(dets)


def test_read_tolerates_trailing_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_detections(path, [_detection("x")])
    path.write_text(path.read_text(encoding="utf-8") + "\n\n  \n", encoding="utf-8")

    loaded = read_detections(path)
    assert len(loaded) == 1
    assert loaded[0].detection_id == "x"


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_atomic_overwrite_no_leftover_tmp(tmp_path: Path) -> None:
    path = tmp_path / "det.jsonl"

    write_detections(path, [_detection("first")])
    write_detections(path, [_detection("second"), _detection("third")])

    loaded = read_detections(path)
    assert [d.detection_id for d in loaded] == ["second", "third"]

    tmp_name = f".{path.name}.tmp"
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == [], f"leftover tmp files: {leftovers}"
    assert not (tmp_path / tmp_name).exists()


# ---------------------------------------------------------------------------
# Ground-truth predicate
# ---------------------------------------------------------------------------


def test_ground_truth_gate() -> None:
    cases = [
        (SOURCE_PERCH, STATUS_PSEUDO, False),
        (SOURCE_YAMNET_PROBE, STATUS_PSEUDO, False),
        (SOURCE_BIRDNET, STATUS_PSEUDO, False),
        (SOURCE_EXPERT, STATUS_CONFIRMED, True),
        (SOURCE_RANGER, STATUS_CORRECTED, True),
        (SOURCE_EXPERT, STATUS_PSEUDO, False),
        (SOURCE_RANGER, STATUS_CONFIRMED, True),
    ]
    for source, status, expected in cases:
        label = Label.now("Turdus merula", 0.5, source, status)
        assert is_ground_truth(label) is expected, (source, status)


# ---------------------------------------------------------------------------
# Prediction lifting
# ---------------------------------------------------------------------------


def test_from_prediction_reuses_fields() -> None:
    pred = Prediction(species="Fringilla coelebs", probability=0.77)
    label = Label.from_prediction(pred, SOURCE_PERCH)

    assert label.species == pred.species
    assert label.probability == pred.probability
    assert label.source == SOURCE_PERCH
    assert label.status == STATUS_PSEUDO
    assert label.ts  # stamped, non-empty
    assert is_ground_truth(label) is False


def test_from_prediction_status_override() -> None:
    pred = Prediction(species="Sylvia atricapilla", probability=0.42)
    label = Label.from_prediction(pred, SOURCE_EXPERT, status=STATUS_CONFIRMED)
    assert label.status == STATUS_CONFIRMED
    assert is_ground_truth(label) is True
