"""export-clips --only-candidates worksheet mode (additive; default unchanged)."""

import csv as _csv
import json
import zipfile

import numpy as np
import soundfile as sf

from faun.cli import _export_clips

SR = 16000


def _write_job(tmp_path):
    job = tmp_path / "job-abc"
    (job / "segments").mkdir(parents=True)
    sig = np.zeros(SR // 2, dtype=np.float32)
    dets = [
        {  # rejected -> candidate
            "detection_id": "d1",
            "trap_id": "A1",
            "source_file": "REC1.wav",
            "segment": {"start_s": 1.0, "duration_s": 2.0},
            "segment_path": "segments/d1.wav",
            "labels": [
                {
                    "species": "__negative__",
                    "probability": 0.2,
                    "source": "model:perch-v2",
                    "status": "rejected",
                    "ts": "2026-06-10T00:00:00Z",
                }
            ],
        },
        {  # pseudo-only -> NOT a candidate
            "detection_id": "d2",
            "trap_id": "A1",
            "source_file": "REC1.wav",
            "segment": {"start_s": 5.0, "duration_s": 2.0},
            "segment_path": "segments/d2.wav",
            "labels": [
                {
                    "species": "Turdus merula",
                    "probability": 0.9,
                    "source": "model:perch",
                    "status": "pseudo",
                    "ts": "2026-06-10T00:00:00Z",
                }
            ],
        },
        {  # rejected + pseudo mix -> candidate
            "detection_id": "d3",
            "trap_id": "A2",
            "source_file": "REC2.wav",
            "segment": {"start_s": 9.0, "duration_s": 2.0},
            "segment_path": "segments/d3.wav",
            "labels": [
                {
                    "species": "Sciurus",
                    "probability": 0.8,
                    "source": "model:perch",
                    "status": "pseudo",
                    "ts": "2026-06-10T00:00:00Z",
                },
                {
                    "species": "__negative__",
                    "probability": 0.1,
                    "source": "model:perch-v2",
                    "status": "rejected",
                    "ts": "2026-06-10T00:00:00Z",
                },
            ],
        },
    ]
    for d in dets:
        sf.write(job / d["segment_path"], sig, SR)
    (job / "detections.jsonl").write_text(
        "\n".join(json.dumps(d) for d in dets) + "\n", encoding="utf-8"
    )
    return job


def test_only_candidates_worksheet(tmp_path):
    job = _write_job(tmp_path)
    out = tmp_path / "worksheet.zip"
    assert _export_clips(job, out, only_candidates=True) == 0

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "clips_index.csv" in names
        assert "segments/d1.wav" in names  # rejected bundled
        assert "segments/d3.wav" in names  # rejected bundled
        assert "segments/d2.wav" not in names  # pseudo-only excluded
        lines = zf.read("clips_index.csv").decode("utf-8").splitlines()

    assert lines[0].split(",") == [
        "detection_id",
        "job_id",
        "trap_id",
        "source_file",
        "start_sec",
        "duration_sec",
        "suggested_species",
        "suggested_source",
        "suggested_probability",
        "corrected_species",
        "notes",
    ]
    rows = list(_csv.DictReader(lines))
    ids = {r["detection_id"] for r in rows}
    assert ids == {"d1", "d3"}
    assert all(r["job_id"] == "job-abc" for r in rows)
    assert all(r["corrected_species"] == "" for r in rows)
    assert all(r["notes"] == "" for r in rows)


def test_default_columns_unchanged(tmp_path):
    job = _write_job(tmp_path)
    out = tmp_path / "clips.zip"
    assert _export_clips(job, out, only_candidates=False) == 0

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        header = zf.read("clips_index.csv").decode("utf-8").splitlines()[0]

    assert header.split(",") == [
        "detection_id",
        "trap_id",
        "source_file",
        "start_sec",
        "duration_sec",
        "suggested_species",
        "suggested_source",
        "suggested_probability",
    ]
    # default mode includes ALL detections' clips
    assert {"segments/d1.wav", "segments/d2.wav", "segments/d3.wav"} <= names
