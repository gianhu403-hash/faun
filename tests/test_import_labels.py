"""import-labels: reverse worksheet CSV -> ranger labels; round-trip + idempotent."""

import csv
import json
import zipfile

import numpy as np
import soundfile as sf

from faun.cli import _export_clips, _export_labels, _import_labels
from faun.detections import (
    SOURCE_RANGER,
    STATUS_CORRECTED,
    is_ground_truth,
    read_detections,
)

SR = 16000


def _make_job(jobs_root, job_id="job-xyz"):
    job = jobs_root / job_id
    (job / "segments").mkdir(parents=True)
    sig = np.zeros(SR // 2, dtype=np.float32)
    det = {
        "detection_id": "det1",
        "trap_id": "A1",
        "source_file": "REC.wav",
        "segment": {"start_s": 3.0, "duration_s": 2.0},
        "segment_path": "segments/det1.wav",
        "labels": [
            {
                "species": "__negative__",
                "probability": 0.1,
                "source": "model:perch-v2",
                "status": "rejected",
                "ts": "2026-06-10T00:00:00Z",
            }
        ],
    }
    sf.write(job / det["segment_path"], sig, SR)
    (job / "detections.jsonl").write_text(json.dumps(det) + "\n", encoding="utf-8")
    return job


def _fill_worksheet(zip_path, csv_out, species):
    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("clips_index.csv").decode("utf-8").splitlines()
    rows = list(csv.DictReader(lines))
    for r in rows:
        r["corrected_species"] = species
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _ranger_labels(job, det_id="det1"):
    dets = read_detections(job / "detections.jsonl")
    target = next(d for d in dets if d.detection_id == det_id)
    return [
        lbl
        for lbl in target.labels
        if lbl.source == SOURCE_RANGER and lbl.status == STATUS_CORRECTED
    ], target


def test_roundtrip_review_to_retrain(tmp_path):
    jobs_root = tmp_path / "jobs"
    job = _make_job(jobs_root)

    ws = tmp_path / "ws.zip"
    _export_clips(job, ws, only_candidates=True)
    filled = tmp_path / "filled.csv"
    _fill_worksheet(ws, filled, species="Sciurus vulgaris")

    assert _import_labels(filled, jobs_root) == 0

    ranger, _ = _ranger_labels(job)
    assert len(ranger) == 1
    assert ranger[0].species == "Sciurus vulgaris"
    assert is_ground_truth(ranger[0])

    out_csv = tmp_path / "labels.csv"
    _export_labels(job, out_csv)
    assert "Sciurus vulgaris" in out_csv.read_text(encoding="utf-8")


def test_idempotent(tmp_path):
    jobs_root = tmp_path / "jobs"
    job = _make_job(jobs_root)
    ws = tmp_path / "ws.zip"
    _export_clips(job, ws, only_candidates=True)
    filled = tmp_path / "filled.csv"
    _fill_worksheet(ws, filled, species="Sciurus vulgaris")

    _import_labels(filled, jobs_root)
    _, t1 = _ranger_labels(job)
    n1 = len(t1.labels)

    _import_labels(filled, jobs_root)
    _, t2 = _ranger_labels(job)
    assert len(t2.labels) == n1  # second run added nothing


def test_blank_and_missing(tmp_path, capsys):
    jobs_root = tmp_path / "jobs"
    job = _make_job(jobs_root)

    filled = tmp_path / "filled.csv"
    with open(filled, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["detection_id", "job_id", "corrected_species"]
        )
        w.writeheader()
        w.writerow(
            {"detection_id": "det1", "job_id": "job-xyz", "corrected_species": ""}
        )  # blank -> skipped
        w.writerow(
            {"detection_id": "nope", "job_id": "job-xyz", "corrected_species": "X"}
        )  # missing det
        w.writerow(
            {
                "detection_id": "det1",
                "job_id": "job-xyz",
                "corrected_species": "Sciurus",
            }
        )  # applied
        w.writerow(
            {"detection_id": "det9", "job_id": "ghost", "corrected_species": "Y"}
        )  # missing job

    assert _import_labels(filled, jobs_root) == 0

    ranger, _ = _ranger_labels(job)
    assert len(ranger) == 1  # only the one good row applied

    out = capsys.readouterr().out
    assert "applied=1" in out
    assert "skipped_blank=1" in out
    assert "missing_det=1" in out
    assert "missing_job=1" in out
