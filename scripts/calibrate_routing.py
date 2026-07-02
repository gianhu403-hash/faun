#!/usr/bin/env python3
"""Calibrate the routing threshold tau_bird from bird vs noise clips.

For each clip, run Perch2Adapter.classify_with_routing → collect p_bird. Report
n_pos/n_neg, mean p_bird per class, ROC-AUC (computed inline; sklearn optional),
and a recommended tau = 5th percentile of the POSITIVE (bird) p_bird. Human
report to stdout; JSON to --out if given.

WHERE TO RUN
    cluster-alex only (needs TensorFlow + a real Perch 2 SavedModel + audio).
    The module import is TF-free — Perch2Adapter and soundfile are imported
    lazily inside main(), so ``--help`` works anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("calibrate_routing")

_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")


def _iter_audio(d: Path):
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
            yield p


def _roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann–Whitney U); no sklearn dependency."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    # Average tied ranks so equal p_bird values do not bias the statistic.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - (counts[i] - 1) / 2.0) for i in range(counts.size)}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[: pos.size].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _collect_p_bird(adapter, clips) -> np.ndarray:
    import soundfile as sf

    vals = []
    for clip in clips:
        wav, sr = sf.read(clip, dtype="float32", always_2d=False)
        result = adapter.classify_with_routing(wav, sr)
        if result.p_bird is not None:
            vals.append(result.p_bird)
        else:
            logger.warning("p_bird uncomputable (mask missing) for %s; skipped", clip)
    return np.asarray(vals, dtype=np.float64)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Calibrate routing tau_bird.")
    ap.add_argument(
        "--birds", required=True, type=Path, help="dir of bird (positive) clips"
    )
    ap.add_argument(
        "--noise", required=True, type=Path, help="dir of noise (negative) clips"
    )
    ap.add_argument("--model", default="perch-v2", choices=["perch-v2"])
    ap.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    for d in (args.birds, args.noise):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 2

    from faun.classification import Perch2Adapter  # lazy: pulls no TF at import

    adapter = Perch2Adapter()
    pos = _collect_p_bird(adapter, list(_iter_audio(args.birds)))
    neg = _collect_p_bird(adapter, list(_iter_audio(args.noise)))
    if pos.size == 0 or neg.size == 0:
        print(
            f"ERROR: need clips with a computable p_bird in both dirs "
            f"(birds={pos.size}, noise={neg.size}); is the eBird-class asset present?",
            file=sys.stderr,
        )
        return 3

    tau = float(np.percentile(pos, 5))
    report = {
        "model": args.model,
        "n_bird": int(pos.size),
        "n_noise": int(neg.size),
        "mean_p_bird_bird": float(pos.mean()),
        "mean_p_bird_noise": float(neg.mean()),
        "roc_auc": _roc_auc(pos, neg),
        "recommended_tau_bird": tau,
        "recommended_tau_rule": "5th percentile of positive p_bird",
    }
    print(json.dumps(report, indent=2))
    print(
        f"\nRecommended FAUN_ROUTING_TAU_BIRD={tau:.4f} "
        f"(AUC={report['roc_auc']:.3f}, bird mean={pos.mean():.3f}, "
        f"noise mean={neg.mean():.3f})"
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
