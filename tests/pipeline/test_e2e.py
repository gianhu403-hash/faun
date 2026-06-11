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
