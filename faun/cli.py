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


if __name__ == "__main__":
    sys.exit(main())
