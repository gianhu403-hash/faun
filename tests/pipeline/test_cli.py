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


def test_retrain_model_routes_to_embedder(tmp_path, monkeypatch):
    """`retrain --model perch-v2` builds the Perch 2 embedder (VC), not yamnet."""
    import faun.cli as cli
    from faun import retraining

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "species,source,status,segment_path\n"
        "Turdus merula,expert:ornithologist,confirmed,segments/a.wav\n",
        encoding="utf-8",
    )

    seen = {}
    sentinel = object()

    def fake_build(name):
        seen["name"] = name
        return sentinel

    def fake_retrain(labels, audio_dir, model, out_path):
        seen["model"] = model
        return {"out_path": str(out_path), "n": len(labels)}

    monkeypatch.setattr(cli, "_build_classifier_by_name", fake_build)
    monkeypatch.setattr(retraining, "retrain_from_labels", fake_retrain)

    rc = main(
        [
            "retrain",
            "--model",
            "perch-v2",
            "--labels",
            str(labels_csv),
            "--out",
            str(tmp_path / "probe.pkl"),
        ]
    )
    assert rc == 0
    assert seen["name"] == "perch-v2"
    assert seen["model"] is sentinel


def test_export_raven_selection_table(tmp_path, capsys):
    """FR-009/SC-E2: export-raven writes a valid Raven TSV with correct times +
    the CURRENT (most recent) label species."""
    import csv
    import json

    job = tmp_path / "job"
    job.mkdir()
    # det1: a model pseudo-label then a human correction -> currentLabel = the
    # correction (last). det2: a single model label.
    det1 = {
        "detection_id": "det1",
        "trap_id": "A1",
        "source_file": "rec1.wav",
        "segment": {"start_s": 2.5, "duration_s": 1.25},
        "segment_path": "segments/det1.wav",
        "labels": [
            {
                "species": "Turdus merula",
                "probability": 0.9,
                "source": "model:perch-v2",
                "status": "pseudo",
                "ts": "t",
            },
            {
                "species": "Erithacus rubecula",
                "probability": None,
                "source": "operator:ranger",
                "status": "corrected",
                "ts": "t",
            },
        ],
    }
    det2 = {
        "detection_id": "det2",
        "trap_id": "A1",
        "source_file": "rec2.wav",
        "segment": {"start_s": 10.0, "duration_s": 2.0},
        "segment_path": "segments/det2.wav",
        "labels": [
            {
                "species": "Parus major",
                "probability": 0.8,
                "source": "model:stub",
                "status": "pseudo",
                "ts": "t",
            },
        ],
    }
    (job / "detections.jsonl").write_text(
        json.dumps(det1) + "\n" + json.dumps(det2) + "\n", encoding="utf-8"
    )

    out = tmp_path / "raven.txt"
    rc = main(["export-raven", "--job", str(job), "--out", str(out)])
    assert rc == 0

    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 2
    # Required Raven columns present.
    assert {"Selection", "Begin Time (s)", "End Time (s)", "Species"} <= set(
        rows[0].keys()
    )
    # det1: current label is the human correction; End = start + duration.
    assert rows[0]["Selection"] == "1"
    assert float(rows[0]["Begin Time (s)"]) == 2.5
    assert float(rows[0]["End Time (s)"]) == 3.75
    assert rows[0]["Species"] == "Erithacus rubecula"
    # det2: the lone model label, times correct.
    assert float(rows[1]["Begin Time (s)"]) == 10.0
    assert float(rows[1]["End Time (s)"]) == 12.0
    assert rows[1]["Species"] == "Parus major"


def test_export_labels_keeps_only_ground_truth(tmp_path, capsys):
    """`export-labels` (VE) emits ONLY human confirmed/corrected labels as a retrain CSV."""
    import csv
    import json

    job = tmp_path / "job"
    job.mkdir()
    # det 1: a model pseudo-label (must be dropped) + a human corrected label (kept).
    # det 2: only a model pseudo-label (entirely dropped).
    det1 = {
        "detection_id": "det1",
        "trap_id": "A1",
        "source_file": "rec1.wav",
        "segment": {"start_s": 1.0, "duration_s": 0.5},
        "segment_path": "segments/det1.wav",
        "labels": [
            {
                "species": "Turdus merula",
                "probability": 0.9,
                "source": "model:perch-v2",
                "status": "pseudo",
                "ts": "t",
            },
            {
                "species": "Fringilla coelebs",
                "probability": None,
                "source": "operator:ranger",
                "status": "corrected",
                "ts": "t",
            },
        ],
    }
    det2 = {
        "detection_id": "det2",
        "trap_id": "A1",
        "source_file": "rec2.wav",
        "segment": {"start_s": 2.0, "duration_s": 0.5},
        "segment_path": "segments/det2.wav",
        "labels": [
            {
                "species": "Parus major",
                "probability": 0.8,
                "source": "model:stub",
                "status": "pseudo",
                "ts": "t",
            },
        ],
    }
    (job / "detections.jsonl").write_text(
        json.dumps(det1) + "\n" + json.dumps(det2) + "\n", encoding="utf-8"
    )

    out = tmp_path / "labels.csv"
    rc = main(["export-labels", "--job", str(job), "--out", str(out)])
    assert rc == 0
    assert "1 ground-truth label(s)" in capsys.readouterr().out

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["species"] == "Fringilla coelebs"
    assert row["source"] == "operator:ranger"
    assert row["status"] == "corrected"
    # segment_path is ABSOLUTE so the CSV feeds `faun retrain` from any location.
    assert Path(row["segment_path"]).is_absolute()
    assert row["segment_path"].endswith("segments/det1.wav")
    # Columns must match what `faun retrain` consumes.
    assert {"species", "source", "status", "segment_path"} <= set(row.keys())


def test_export_labels_then_retrain_resolves_clips(tmp_path):
    """End-to-end: export-labels -> retrain resolves the exported clips (closed loop).

    The exported CSV lives in a DIFFERENT directory than the job, proving the
    absolute segment_path lets retrain resolve clips regardless of --audio-dir.
    """
    import csv
    import json

    import numpy as np
    import soundfile as sf

    from faun import retraining

    job = tmp_path / "job"
    (job / "segments").mkdir(parents=True)
    dets = []
    for sp, did in [("Turdus merula", "d0"), ("Fringilla coelebs", "d1")]:
        sf.write(
            job / "segments" / f"{did}.wav", np.zeros(16000, dtype=np.float32), 16000
        )
        dets.append(
            {
                "detection_id": did,
                "trap_id": "A1",
                "source_file": "r.wav",
                "segment": {"start_s": 0.0, "duration_s": 1.0},
                "segment_path": f"segments/{did}.wav",
                "labels": [
                    {
                        "species": sp,
                        "probability": None,
                        "source": "operator:ranger",
                        "status": "corrected",
                        "ts": "t",
                    }
                ],
            }
        )
    (job / "detections.jsonl").write_text(
        "\n".join(json.dumps(d) for d in dets) + "\n", encoding="utf-8"
    )

    # Export to a directory UNRELATED to the job (tests absolute-path resolution).
    csv_out = tmp_path / "elsewhere" / "labels.csv"
    assert main(["export-labels", "--job", str(job), "--out", str(csv_out)]) == 0
    rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(Path(r["segment_path"]).is_absolute() for r in rows)

    class _FakeModel:
        def __init__(self):
            self.n = 0

        def embed(self, wav, sr):
            self.n += 1
            return np.full(8, float(self.n), dtype=np.float32)

    # audio_dir intentionally unrelated; the absolute segment_path must resolve.
    metrics = retraining.retrain_from_labels(
        rows,
        audio_dir=tmp_path / "unrelated",
        model=_FakeModel(),
        out_path=tmp_path / "probe.pkl",
    )
    assert metrics["n"] == 2  # both clips resolved + embedded (loop is closed)
    assert "out_path" in metrics


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
