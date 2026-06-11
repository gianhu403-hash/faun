"""CLI tests — run_pipeline is patched; the real chain is never invoked."""

from __future__ import annotations

from pathlib import Path

import faun.api as api
from faun.cli import main


def test_process_prints_csv_path(tmp_path, monkeypatch, capsys):
    src = tmp_path / "A1"
    src.mkdir()

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        results = Path(job_dir) / "results.csv"
        results.write_text("track,start_sec\nA1,0.0\n", encoding="utf-8")
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    rc = main(["process", str(src)])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    printed = Path(out)
    assert printed.exists()
    assert printed.name == "results.csv"


def test_process_honours_out_flag(tmp_path, monkeypatch, capsys):
    src = tmp_path / "A1"
    src.mkdir()
    out_path = tmp_path / "custom" / "out.csv"

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        results = Path(job_dir) / "results.csv"
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text("track,start_sec\nA1,0.0\n", encoding="utf-8")
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    rc = main(["process", str(src), "--out", str(out_path)])
    assert rc == 0

    printed = Path(capsys.readouterr().out.strip())
    assert printed == out_path
    assert out_path.exists()


def test_process_passes_source_dir(tmp_path, monkeypatch, capsys):
    src = tmp_path / "A1"
    src.mkdir()
    seen = {}

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        seen["source"] = source_path
        results = Path(job_dir) / "results.csv"
        results.write_text("x\n", encoding="utf-8")
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)
    main(["process", str(src)])
    assert seen["source"] == str(src)
