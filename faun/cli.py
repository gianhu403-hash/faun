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
    retrain.add_argument("--model", default="yamnet", choices=["yamnet"])
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
        choices=["perch", "yamnet"],
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
    evalsp.add_argument("--embedder", default="perch", choices=["perch", "yamnet"])
    evalsp.add_argument("--seed", type=int, default=42)
    evalsp.add_argument(
        "--real",
        action="store_true",
        help="dataset is real iNatSounds (cluster) -> emit a REAL species metric; "
        "default keeps the honest SYNTHETIC tag",
    )

    args = parser.parse_args(argv)

    if args.command == "process":
        # Imported here so the patch target is faun.api.run_pipeline and the
        # CLI stays cheap to import.
        from faun.api import run_pipeline

        src = Path(args.dir)
        out = Path(args.out) if args.out else src / "results.csv"
        job_dir = out.parent
        results = run_pipeline(job_dir, str(src))
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

        if args.model == "yamnet":
            from faun.classification import YAMNetAdapter

            model = YAMNetAdapter()
        else:  # pragma: no cover - argparse choices already constrain this
            raise SystemExit(f"unknown model: {args.model}")

        metrics = retraining.retrain_from_labels(
            labels, audio_dir=audio_dir, model=model, out_path=Path(args.out)
        )
        print(metrics)
        return 0

    if args.command == "export-clips":
        return _export_clips(Path(args.job), Path(args.out))

    if args.command == "batch-label":
        return _batch_label(args)

    if args.command == "fetch-dataset":
        return _fetch_dataset(Path(args.root))

    if args.command == "eval-species":
        return _eval_species(args)

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


def _build_classifier_by_name(name: str):
    """Map a model name to a classifier adapter (heavy ML deps imported lazily)."""
    from faun.classification import StubAdapter

    if name == "stub":
        return StubAdapter()
    if name == "perch":
        from faun.classification import PerchAdapter

        return PerchAdapter()
    if name == "birdnet":
        from faun.classification import BirdNETAdapter

        return BirdNETAdapter()
    if name == "yamnet":
        from faun.classification import YAMNetAdapter

        return YAMNetAdapter()
    raise SystemExit(f"unknown model: {name}")


def _build_embedder(name: str):
    """Map an embedder name to an Embedder adapter (heavy ML deps imported lazily)."""
    from faun.embeddings import PerchEmbedder, YamnetEmbedder

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
        Path(args.archive),
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


if __name__ == "__main__":
    sys.exit(main())
