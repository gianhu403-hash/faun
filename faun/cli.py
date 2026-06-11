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

    return 1


if __name__ == "__main__":
    sys.exit(main())
