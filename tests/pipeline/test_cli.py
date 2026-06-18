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


# ---------------------------------------------------------------------------
# Wave-2 integration subcommands: fetch-dataset / batch-label / eval-species
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_fetch_dataset_on_mini_fixture(capsys):
    """fetch-dataset validates a real iNatSounds tree (TF-free) and reports vocab."""
    root = _FIXTURES / "inatsounds_mini"
    rc = main(["fetch-dataset", "--root", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "'n_species': 3" in out
    assert "Turdus_merula" in out


def test_batch_label_stub_archive(tmp_path, capsys):
    """batch-label with the TF-free Stub classifier writes a real detections.jsonl."""
    from faun.detections import read_detections

    archive = _FIXTURES / "traps_mini"
    out = tmp_path / "detections.jsonl"
    rc = main(
        [
            "batch-label",
            "--archive",
            str(archive),
            "--out",
            str(out),
            "--models",
            "stub",
        ]
    )
    assert rc == 0
    assert out.exists()
    dets = read_detections(out)
    # The stub yields >=1 label per detection, all sourced model:stub (pseudo).
    assert dets, "expected at least one detection from the stub run"
    assert all(
        lbl.source == "model:stub" and lbl.status == "pseudo"
        for d in dets
        for lbl in d.labels
    )


def test_eval_species_dispatch(monkeypatch, capsys):
    """eval-species routes load_probe -> embed -> species_eval (real eval, TF-free).

    The heavy embed step (_embed_split) and probe load are stubbed; species_eval
    itself runs for real on synthetic embeddings so the dispatch wiring is exercised.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    import faun.cli as cli
    from faun import retraining

    rng = np.random.default_rng(0)
    # Two separable clusters per class so the probe actually learns.
    X = np.vstack([rng.normal(0, 0.1, (8, 4)), rng.normal(5, 0.1, (8, 4))])
    y = np.array(["a"] * 8 + ["b"] * 8)
    clf = LogisticRegression(max_iter=500).fit(X, y)

    monkeypatch.setattr(retraining, "load_probe", lambda path: clf)
    monkeypatch.setattr(cli, "_embed_split", lambda records, embedder: (X, y))
    monkeypatch.setattr(cli, "_build_embedder", lambda name: object())

    root = _FIXTURES / "inatsounds_mini"

    # --real -> non-SYNTHETIC provenance (the cluster path).
    rc = main(
        [
            "eval-species",
            "--probe",
            "/nonexistent.pkl",
            "--dataset",
            str(root),
            "--real",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "macro_f1" in out
    assert "SYNTHETIC" not in out

    # Honest default (no --real) -> result is tagged SYNTHETIC.
    rc = main(["eval-species", "--probe", "/nonexistent.pkl", "--dataset", str(root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC — not a species metric" in out


# ---------------------------------------------------------------------------
# Wave-3 completeness: finetune dispatch + process-<url> passthrough
# ---------------------------------------------------------------------------


def test_finetune_dispatch_forwards_args(tmp_path, monkeypatch):
    """`faun finetune` wires argparse (incl. --no-amp/--no-class-weight) to finetune()."""
    import faun.training as training

    seen = {}

    def spy(dataset_root, **kw):
        seen["dataset_root"] = dataset_root
        seen.update(kw)
        return {"provenance": "SYNTHETIC — not a species metric", "best_epoch": 0}

    monkeypatch.setattr(training, "finetune", spy)

    out = tmp_path / "ckpt"
    rc = main(
        [
            "finetune",
            "--dataset",
            str(_FIXTURES / "inatsounds_mini"),
            "--out",
            str(out),
            "--model",
            "passt",
            "--no-amp",
            "--no-class-weight",
            "--epochs",
            "1",
        ]
    )
    assert rc == 0
    assert seen["dataset_root"] == str(_FIXTURES / "inatsounds_mini")
    assert seen["out"] == str(out)
    assert seen["model"] == "passt"
    assert seen["amp"] is False  # --no-amp store_false
    assert seen["class_weight"] is False  # --no-class-weight store_false
    assert seen["epochs"] == 1


def test_process_url_passthrough_no_path_mangle(monkeypatch):
    """`process <url>`: the URL reaches run_pipeline verbatim (// intact); job_dir isolated, not cwd."""
    seen = {}

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        seen["job_dir"] = Path(job_dir)
        seen["source"] = source_path
        results = Path(job_dir) / "results.csv"
        results.write_text("x\n", encoding="utf-8")
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    rc = main(["process", "https://disk.yandex.ru/d/KEY/A1"])
    assert rc == 0
    # The // must survive — this is the whole point of the P0 fix.
    assert seen["source"] == "https://disk.yandex.ru/d/KEY/A1"
    assert "https://" in seen["source"]
    # URL jobs get an isolated fresh dir, not the operator's cwd.
    assert seen["job_dir"] != Path.cwd()
