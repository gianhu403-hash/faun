"""End-to-end seam test: the REAL run_pipeline chain, no mocks.

Every other api/cli test patches ``run_pipeline`` — this is the single place
where the cross-wave contracts (ingest -> ordering -> segmentation ->
classification -> output) are actually executed together, so signature drift
in any module fails here instead of in production.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from faun.api import run_pipeline
from faun.output import COLUMNS

SR = 48_000


def _make_trap_dir(root: Path, trap_id: str = "A1") -> Path:
    """Synthetic trap folder: one stereo 48k WAV with a loud burst + info.txt."""
    trap = root / trap_id
    trap.mkdir(parents=True)
    t = np.linspace(0, 6, 6 * SR, endpoint=False)
    sig = 0.005 * np.random.default_rng(7).standard_normal((6 * SR, 2))
    burst = np.sin(2 * np.pi * 3000 * t) * ((t > 2.5) & (t < 3.0))
    sig[:, 0] += burst
    sig[:, 1] += burst
    sf.write(trap / "REC_20260610_213000.wav", sig, SR)
    (trap / "info.txt").write_text(
        "date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel\n"
        "2026-06-10,21:30:00,32.95,60.55,3.9,14.2,71,"
        "REC_20260610_213000.wav,48000,auto,stereo\n",
        encoding="utf-8",
    )
    return trap


def test_run_pipeline_real_chain(tmp_path: Path) -> None:
    _make_trap_dir(tmp_path / "data")
    job_dir = tmp_path / "job"

    results = run_pipeline(job_dir, str(tmp_path / "data"))

    assert results.exists()
    with results.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # The burst must be detected and classified by the default StubAdapter.
    assert rows, "burst at 2.5s must yield at least one prediction row"
    assert list(rows[0].keys()) == list(COLUMNS)
    assert rows[0]["track"] == "A1"
    assert 1.5 <= float(rows[0]["start_sec"]) <= 3.5
    species = {r["species"] for r in rows}
    assert "Turdus merula" in species  # StubAdapter's fixture prediction

    # Sidecar carries single-trap provenance.
    sidecar = json.loads((job_dir / "results_meta.json").read_text(encoding="utf-8"))
    assert sidecar["trap_id"] == "A1"
    assert sidecar["files"] == ["REC_20260610_213000.wav"]

    # --- Phase-2: detections.jsonl + real clips ---------------------------
    from faun.detections import read_detections

    jsonl = job_dir / "detections.jsonl"
    assert jsonl.exists()
    dets = read_detections(jsonl)
    assert len(dets) >= 1, "burst must persist at least one detection"

    seg_dir = job_dir / "segments"
    clips = list(seg_dir.glob("*.wav"))
    assert clips, "segments/ must contain the clip(s)"
    assert len(clips) == len(dets)

    # The original waveform, to compare clip content against the slice.
    original, orig_sr = sf.read(
        tmp_path / "data" / "A1" / "REC_20260610_213000.wav",
        dtype="float64",
        always_2d=False,
    )

    det = dets[0]
    clip_path = job_dir / det.segment_path
    info = sf.info(str(clip_path))
    assert info.samplerate == SR  # original sr preserved (48000)

    clip, clip_sr = sf.read(clip_path, dtype="float64", always_2d=False)
    assert clip_sr == SR
    # Clip length / sr is within tolerance of the detection duration_s.
    assert abs(len(clip) / clip_sr - det.segment.duration_s) < 0.05

    # Clip content equals the corresponding slice of the original waveform.
    start = round(det.segment.start_s * SR)
    end = round((det.segment.start_s + det.segment.duration_s) * SR)
    expected = original[start:end]
    assert clip.shape == expected.shape
    assert np.allclose(clip, expected, atol=1e-4)


def test_run_pipeline_multi_trap_sidecar_is_honest(tmp_path: Path) -> None:
    _make_trap_dir(tmp_path / "data", "A1")
    _make_trap_dir(tmp_path / "data", "A2")
    job_dir = tmp_path / "job"

    run_pipeline(job_dir, str(tmp_path / "data"))

    sidecar = json.loads((job_dir / "results_meta.json").read_text(encoding="utf-8"))
    # A mixed batch must not claim the first trap's identity/coords.
    assert sidecar["trap_id"] == "multi"
    assert sidecar["lat"] is None and sidecar["lon"] is None
    assert len(sidecar["files"]) == 2


def test_run_pipeline_writes_species_presence(tmp_path: Path) -> None:
    """FR-005/SC-004: species_presence.json is valid and counts match the jsonl."""
    _make_trap_dir(tmp_path / "data")
    job_dir = tmp_path / "job"

    run_pipeline(job_dir, str(tmp_path / "data"))

    from faun.detections import read_detections

    presence_path = job_dir / "species_presence.json"
    assert presence_path.is_file()
    presence = json.loads(presence_path.read_text(encoding="utf-8"))
    dets = read_detections(job_dir / "detections.jsonl")

    # Group totals sum to the number of detections (honest count, SC-004).
    total = sum(g["n_detections"] for g in presence["groups"])
    assert total == len(dets)
    # StubAdapter's argmax label is Turdus merula -> it must appear for trap A1.
    species = {s["species"] for g in presence["groups"] for s in g["species"]}
    assert "Turdus merula" in species
    assert all(g["trap_id"] == "A1" for g in presence["groups"])


def test_run_pipeline_allowlist_off_is_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    """SC-002 golden diff: a configured allow-list the mask can't activate (Stub
    has no vocabulary) leaves results.csv byte-for-byte identical to baseline."""
    from faun.settings import get_settings

    _make_trap_dir(tmp_path / "data")

    # Baseline: no allow-list at all (the default prod-safe path).
    monkeypatch.delenv("FAUN_SPECIES_ALLOWLIST", raising=False)
    get_settings.cache_clear()
    baseline_csv = run_pipeline(tmp_path / "job_off", str(tmp_path / "data"))
    baseline_bytes = baseline_csv.read_bytes()

    # Allow-list configured, but StubAdapter exposes no vocab -> mask is a no-op.
    monkeypatch.setenv("FAUN_SPECIES_ALLOWLIST", "default")
    get_settings.cache_clear()
    masked_csv = run_pipeline(tmp_path / "job_on", str(tmp_path / "data"))
    assert masked_csv.read_bytes() == baseline_bytes


def test_run_pipeline_active_mask_restricts_species(tmp_path: Path) -> None:
    """SC-002 (ON): an active mask keeps only allow-listed species in the CSV."""
    from faun.classification import MaskedClassifier, Prediction

    _make_trap_dir(tmp_path / "data")
    job_dir = tmp_path / "job"

    vocab = ["Turdus merula", "Zonotrichia leucophrys", "Parus major"]

    class _RichClf:
        def classify(self, segment, sr):
            return [
                Prediction("Zonotrichia leucophrys", 0.95),  # out-of-region
                Prediction("Turdus merula", 0.80),
            ]

    masked = MaskedClassifier(
        _RichClf(), ["Turdus merula", "Parus major"], vocab_provider=lambda: vocab
    )
    results = run_pipeline(job_dir, str(tmp_path / "data"), classifier=masked)

    with results.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "burst must still produce rows"
    species = {r["species"] for r in rows}
    assert species == {"Turdus merula"}  # out-of-region species dropped


def test_run_pipeline_silence_yields_empty_csv(tmp_path: Path) -> None:
    trap = tmp_path / "data" / "A1"
    trap.mkdir(parents=True)
    quiet = 0.0005 * np.random.default_rng(3).standard_normal((3 * SR, 2))
    sf.write(trap / "REC_20260610_220000.wav", quiet, SR)

    results = run_pipeline(tmp_path / "job", str(tmp_path / "data"))

    with results.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == list(COLUMNS)
        assert list(reader) == []  # header only — no events in silence

    # Phase-2: detections.jsonl exists but is empty (0 lines), no clips.
    job_dir = tmp_path / "job"
    jsonl = job_dir / "detections.jsonl"
    assert jsonl.exists()
    assert jsonl.read_text(encoding="utf-8") == ""
    assert not list((job_dir / "segments").glob("*.wav"))
