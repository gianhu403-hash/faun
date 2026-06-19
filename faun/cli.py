"""Faun CLI — ``python -m faun.cli process <dir> [--out results.csv]``.

Runs the pipeline synchronously via ``faun.api.run_pipeline`` and prints the
path to the written CSV. Tests patch ``run_pipeline``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="faun", description="Faun pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser("process", help="run the pipeline on a trap folder")
    process.add_argument("dir", help="source directory (one folder per trap)")
    process.add_argument(
        "--out",
        default=None,
        help="output CSV path (default: <dir>/results.csv)",
    )

    retrain = sub.add_parser(
        "retrain", help="retrain the species probe from human labels"
    )
    retrain.add_argument("--labels", required=True, help="CSV of reviewed labels")
    retrain.add_argument(
        "--model",
        default="yamnet",
        choices=["yamnet", "perch", "perch-v2"],
        help="embedding backbone for the probe (default: yamnet)",
    )
    retrain.add_argument("--out", required=True, help="output probe path (.pkl)")
    retrain.add_argument(
        "--audio-dir",
        default=None,
        help="directory holding the clips (default: labels CSV parent)",
    )

    export = sub.add_parser(
        "export-clips", help="export real .wav clips + index for ornithologists"
    )
    export.add_argument("--job", required=True, help="job dir with detections.jsonl")
    export.add_argument("--out", required=True, help="output ZIP path")

    exlabels = sub.add_parser(
        "export-labels",
        help="export human ground-truth labels from a job -> retrain CSV",
    )
    exlabels.add_argument("--job", required=True, help="job dir with detections.jsonl")
    exlabels.add_argument(
        "--out", required=True, help="output labels CSV (feeds `faun retrain`)"
    )

    blabel = sub.add_parser(
        "batch-label",
        help="multi-model pseudo-labeling (Perch+BirdNET) of a trap archive",
    )
    blabel.add_argument("--archive", required=True, help="trap archive dir (A1..A5)")
    blabel.add_argument("--out", required=True, help="output detections.jsonl path")
    blabel.add_argument("--emb-out", default=None, help="optional embeddings.npz path")
    blabel.add_argument(
        "--embedder",
        default="perch",
        choices=["perch", "perch-v2", "yamnet"],
        help="embedder for --emb-out (default: perch)",
    )
    blabel.add_argument(
        "--models",
        default="perch,birdnet",
        help="comma list of classifiers (perch,birdnet,yamnet,stub)",
    )

    fetch = sub.add_parser(
        "fetch-dataset", help="locate + validate an iNatSounds dataset tree"
    )
    fetch.add_argument(
        "--root", required=True, help="dataset root (root/<species>/<clip>)"
    )

    evalsp = sub.add_parser(
        "eval-species",
        help="evaluate a species probe on a labeled dataset (real metric on cluster)",
    )
    evalsp.add_argument("--probe", required=True, help="pickled probe (.pkl)")
    evalsp.add_argument("--dataset", required=True, help="iNatSounds dataset root")
    evalsp.add_argument(
        "--embedder", default="perch", choices=["perch", "perch-v2", "yamnet"]
    )
    evalsp.add_argument("--seed", type=int, default=42)
    evalsp.add_argument(
        "--real",
        action="store_true",
        help="dataset is real iNatSounds (cluster) -> emit a REAL species metric; "
        "default keeps the honest SYNTHETIC tag",
    )

    finetune = sub.add_parser(
        "finetune",
        help="REAL transformer fine-tune on iNatSounds (cluster GPU; faun.training)",
    )
    finetune.add_argument(
        "--dataset", required=True, help="iNatSounds root (root/<species>/<clip>)"
    )
    finetune.add_argument("--model", default="passt", choices=["passt", "ast", "beats"])
    finetune.add_argument(
        "--out", required=True, help="output checkpoint dir (meta.json + weights.pt)"
    )
    finetune.add_argument("--epochs", type=int, default=15)
    finetune.add_argument("--batch-size", type=int, default=16)
    finetune.add_argument("--lr", type=float, default=3e-4)
    finetune.add_argument("--device", default="auto")
    finetune.add_argument("--amp", action="store_true", default=True)
    finetune.add_argument("--no-amp", dest="amp", action="store_false")
    finetune.add_argument("--grad-accum", type=int, default=2)
    finetune.add_argument("--freeze-epochs", type=int, default=3)
    finetune.add_argument("--patience", type=int, default=4)
    finetune.add_argument("--class-weight", action="store_true", default=True)
    finetune.add_argument(
        "--no-class-weight", dest="class_weight", action="store_false"
    )
    finetune.add_argument("--seed", type=int, default=42)
    finetune.add_argument("--resume", default=None, help="resume from checkpoint dir")

    args = parser.parse_args(argv)

    if args.command == "process":
        # Imported here so the patch target is faun.api.run_pipeline and the
        # CLI stays cheap to import.
        from faun.api import run_pipeline

        # The source may be a local dir OR a URL / Yandex.Disk share (resolved
        # inside run_pipeline). A URL has no local parent for the default --out,
        # so fall back to cwd; pass the RAW arg so the // in a URL survives.
        src_arg = args.dir
        is_url = src_arg.lower().startswith(("http://", "https://"))
        if args.out:
            out = Path(args.out)
            job_dir = out.parent
        elif is_url:
            # A URL source has no local parent dir; isolate the job in a fresh
            # per-run dir (mirrors the API's jobs_root/<id>/) so results.csv +
            # segments/ + the extracted _source/ don't scatter across the
            # operator's cwd or collide between runs.
            import tempfile

            job_dir = Path(tempfile.mkdtemp(prefix="faun-job-"))
            out = job_dir / "results.csv"
        else:
            out = Path(src_arg) / "results.csv"
            job_dir = out.parent
        results = run_pipeline(job_dir, src_arg)
        # Honour an explicit --out path if run_pipeline wrote elsewhere.
        results = Path(results)
        if args.out and results != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(results.read_bytes())
            results = out
        print(results)
        return 0

    if args.command == "retrain":
        # Heavy deps (the embedder pulls TF lazily) live inside the branch.
        import csv

        from faun import retraining

        labels_path = Path(args.labels)
        with open(labels_path, newline="", encoding="utf-8") as fh:
            labels = list(csv.DictReader(fh))

        audio_dir = Path(args.audio_dir) if args.audio_dir else labels_path.parent

        # retrain_from_labels is embedder-agnostic — it only calls model.embed.
        # All three choices (yamnet/perch/perch-v2) expose embed(); argparse
        # choices already constrain the value to an embed-capable adapter.
        model = _build_classifier_by_name(args.model)

        metrics = retraining.retrain_from_labels(
            labels, audio_dir=audio_dir, model=model, out_path=Path(args.out)
        )
        print(metrics)
        return 0

    if args.command == "export-clips":
        return _export_clips(Path(args.job), Path(args.out))

    if args.command == "export-labels":
        return _export_labels(Path(args.job), Path(args.out))

    if args.command == "batch-label":
        return _batch_label(args)

    if args.command == "fetch-dataset":
        return _fetch_dataset(Path(args.root))

    if args.command == "eval-species":
        return _eval_species(args)

    if args.command == "finetune":
        return _finetune(args)

    return 1


def _export_clips(job_dir: Path, out_path: Path) -> int:
    """Bundle each detection's real .wav clip plus an index CSV into a ZIP.

    Reads ``<job_dir>/detections.jsonl`` (one JSON object per line, parsed
    generically per the detection contract — no import of ``faun.detections``)
    and writes ``out_path`` containing every existing ``segment_path`` clip and
    a ``clips_index.csv`` summarising the top label per detection. This is the
    ornithologist hand-off: real audio clips, never spectrograms.
    """
    import csv
    import io
    import json
    import zipfile

    jsonl = job_dir / "detections.jsonl"
    detections = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                detections.append(json.loads(line))

    index = io.StringIO()
    writer = csv.writer(index)
    writer.writerow(
        [
            "detection_id",
            "trap_id",
            "source_file",
            "start_sec",
            "duration_sec",
            "suggested_species",
            "suggested_source",
            "suggested_probability",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for det in detections:
            segment = det.get("segment") or {}
            labels = det.get("labels") or []
            top = labels[0] if labels else {}
            writer.writerow(
                [
                    det.get("detection_id", ""),
                    det.get("trap_id", ""),
                    det.get("source_file", ""),
                    segment.get("start_s", ""),
                    segment.get("duration_s", ""),
                    top.get("species", ""),
                    top.get("source", ""),
                    top.get("probability", ""),
                ]
            )
            seg_path = det.get("segment_path")
            if seg_path:
                clip = job_dir / seg_path
                if clip.exists():
                    zf.write(clip, seg_path)
        zf.writestr("clips_index.csv", index.getvalue())

    print(out_path)
    return 0


def _export_labels(job_dir: Path, out_path: Path) -> int:
    """Export a job's HUMAN ground-truth labels to a retrain-ready CSV.

    Reads ``<job_dir>/detections.jsonl`` and, for each detection, keeps only the
    labels that pass the canonical ground-truth gate
    (:func:`faun.detections.is_ground_truth` — human source expert/ranger AND
    status confirmed/corrected; model pseudo-labels are dropped). Each surviving
    label becomes one CSV row carrying the columns ``faun retrain`` consumes
    (``species``/``source``/``status``/``segment_path``). This closes the loop so
    operator review labels feed retraining with no code change.

    Importing ``faun.detections`` here is deliberate: the ground-truth predicate
    must stay single-homed (unlike ``export-clips`` which is provenance-agnostic).
    The module is stdlib-light (no TF).
    """
    import csv
    import json

    from faun.detections import is_ground_truth

    jsonl = job_dir / "detections.jsonl"
    fieldnames = [
        "species",
        "source",
        "status",
        "segment_path",
        "source_file",
        "start_sec",
        "duration_sec",
        "detection_id",
    ]
    rows: list[dict] = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            det = json.loads(line)
            seg = det.get("segment") or {}
            for lbl in det.get("labels") or []:
                if not is_ground_truth(lbl):
                    continue
                rows.append(
                    {
                        "species": lbl.get("species", ""),
                        "source": lbl.get("source", ""),
                        "status": lbl.get("status", ""),
                        "segment_path": det.get("segment_path", ""),
                        "source_file": det.get("source_file", ""),
                        "start_sec": seg.get("start_s", ""),
                        "duration_sec": seg.get("duration_s", ""),
                        "detection_id": det.get("detection_id", ""),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{out_path} ({len(rows)} ground-truth label(s))")
    return 0


def _build_classifier_by_name(name: str):
    """Map a model name to a classifier adapter (heavy ML deps imported lazily)."""
    from faun.classification import StubAdapter

    if name == "stub":
        return StubAdapter()
    if name == "perch":
        from faun.classification import PerchAdapter

        return PerchAdapter()
    if name == "perch-v2":
        from faun.classification import Perch2Adapter

        return Perch2Adapter()
    if name == "birdnet":
        from faun.classification import BirdNETAdapter

        return BirdNETAdapter()
    if name == "yamnet":
        from faun.classification import YAMNetAdapter

        return YAMNetAdapter()
    raise SystemExit(f"unknown model: {name}")


def _build_embedder(name: str):
    """Map an embedder name to an Embedder adapter (heavy ML deps imported lazily)."""
    from faun.embeddings import Perch2Embedder, PerchEmbedder, YamnetEmbedder

    if name == "perch-v2":
        return Perch2Embedder()
    return PerchEmbedder() if name == "perch" else YamnetEmbedder()


def _batch_label(args) -> int:
    """Run multi-model pseudo-labeling over a trap archive (faun.labeling.batch_label).

    BirdNET pseudo-labels are inventory/prioritization only — the NC+SA gate in
    ``faun.labeling.training_candidates`` keeps them out of any training set.
    """
    from faun import labeling

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {name: _build_classifier_by_name(name) for name in names}
    embedder = _build_embedder(args.embedder) if args.emb_out else None
    summary = labeling.batch_label(
        args.archive,  # raw str: local dir | http(s) zip | Yandex.Disk share
        models,
        Path(args.out),
        emb_out=Path(args.emb_out) if args.emb_out else None,
        embedder=embedder,
    )
    print(summary)
    return 0


def _fetch_dataset(root: Path) -> int:
    """Validate an iNatSounds tree and print its manifest/vocabulary summary."""
    from faun.datasets import iNatSoundsDataset

    ds = iNatSoundsDataset(root)
    manifest = ds.manifest()
    vocab = ds.vocab()
    print(
        {
            "root": str(root),
            "n_records": len(manifest),
            "n_species": len(vocab),
            "species": sorted(vocab),
        }
    )
    return 0


def _embed_split(records, embedder):
    """Embed iNatRecord clips -> ``(X, y)``. Heavy: pulls the embedder (lazy TF)."""
    import numpy as np
    import soundfile as sf

    from faun.embeddings import embed_batch

    clips = []
    species: list[str] = []
    for rec in records:
        waveform, sr = sf.read(rec.path, dtype="float32", always_2d=False)
        clips.append((np.asarray(waveform), int(sr)))
        species.append(rec.species)
    return embed_batch(clips, embedder), np.asarray(species)


def _eval_species(args) -> int:
    """Evaluate a saved probe on a dataset's val split.

    Heavy path (embedder pulls TF) — runs on the cluster. Pass ``--real`` only for
    the true iNatSounds dataset to get a non-SYNTHETIC provenance; without it the
    result is tagged ``SYNTHETIC — not a species metric`` (honesty default).
    """
    from faun import retraining
    from faun.datasets import iNatSoundsDataset

    clf = retraining.load_probe(Path(args.probe))
    ds = iNatSoundsDataset(Path(args.dataset))
    _train, val = ds.split(args.seed)
    embedder = _build_embedder(args.embedder)
    X, y = _embed_split(val, embedder)
    # Honest by default: only --real (true cluster iNatSounds) yields a non-SYNTHETIC
    # provenance tag — a toy/mini tree must never be reported as a species metric.
    metrics = retraining.species_eval(clf, X, y, synthetic=not args.real)
    print(metrics)
    return 0


def _finetune(args) -> int:
    """Run the REAL transformer fine-tune on iNatSounds (faun.training).

    Heavy deps (torch, hear21passt) are imported lazily inside ``finetune()``; the
    CLI import stays cheap. Real species numbers only exist after the cluster run
    (``scripts/finetune_inatsounds.sh``) — a local/synthetic run is not a metric.
    """
    from faun.training import finetune

    summary = finetune(
        args.dataset,
        model=args.model,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        amp=args.amp,
        grad_accum=args.grad_accum,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        class_weight=args.class_weight,
        seed=args.seed,
        resume=args.resume,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
