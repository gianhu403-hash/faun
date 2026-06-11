"""E1 — BirdNET baseline on a labelled bird-detection subset (CPU).

Goal
----
Sanity-check BirdNET as an off-the-shelf "is there a bird?" detector on an
open, weakly-labelled subset (freefield1010 or warblrb10k: WAV + binary
``hasbird`` 0/1). We run BirdNET over N seeded samples, take the max
per-clip species confidence as the bird-presence score, and report ROC-AUC
plus precision/recall at a confidence threshold.

This is an *inventory / sanity* baseline, not a species-accuracy claim:
BirdNET is CC BY-NC-SA, so its head cannot taint a fine-tuned model — we
only measure how well it separates bird vs no-bird out of the box. CPU only
(birdnetlib + tensorflow-cpu, image faun-ml-cpu).

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
cfg keys: data_root, raw180, datasets, hf_cache, results_dir.
Wrapper: experiments.wrappers.birdnet (analyze_file -> [(start,end,name,conf)]);
missing wrapper/model/data -> graceful skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiments import common

logger = logging.getLogger("faun.experiments.e1")

MODEL = "BirdNET"
METRIC = "roc_auc"
DEFAULT_N = 400
DEFAULT_SEED = 1337
DEFAULT_THRESH = 0.1  # birdnetlib default min_conf; bird-presence decision point
DATASET_DIRS = ("freefield1010", "ff1010bird", "warblrb10k", "warblrb10k_public")


def _find_dataset(cfg: dict) -> Path | None:
    base = Path(cfg["datasets"])
    for name in DATASET_DIRS:
        p = base / name
        if p.is_dir():
            return p
    return None


def _find_label_csv(ds_dir: Path) -> Path | None:
    prefer = ("ff1010bird_metadata", "warblrb10k_public_metadata", "labels", "metadata")
    csvs = sorted(ds_dir.rglob("*.csv"))
    for key in prefer:
        for c in csvs:
            if key in c.name.lower():
                return c
    return csvs[0] if csvs else None


def run(cfg: dict) -> dict:
    import numpy as np

    seed = int(cfg.get("seed", DEFAULT_SEED))
    n = int(cfg.get("n_samples", DEFAULT_N))
    thresh = float(cfg.get("birdnet_thresh", DEFAULT_THRESH))

    ds_dir = _find_dataset(cfg)
    if ds_dir is None:
        return {
            "skip": "no labelled bird-detection dataset under datasets/ "
            "(freefield1010|warblrb10k)"
        }
    dataset = ds_dir.name

    label_csv = _find_label_csv(ds_dir)
    if label_csv is None:
        return {"skip": f"no label CSV in {dataset}"}
    try:
        labels = common.read_bird_labels(label_csv)
    except (KeyError, ValueError) as exc:
        return {
            "skip": f"label CSV {label_csv.name} not freefield/warblr schema: {exc}"
        }
    if not labels:
        return {"skip": f"empty labels in {label_csv.name}"}

    files = common.sample_files(ds_dir, n=n, seed=seed)
    labelled = [(p, labels[p.stem]) for p in files if p.stem in labels]
    if len(labelled) < 20:
        return {"skip": f"only {len(labelled)} labelled files matched in {dataset}"}

    try:
        from experiments.wrappers import birdnet
    except Exception as exc:
        return {"skip": f"birdnet wrapper unavailable: {type(exc).__name__}: {exc}"}

    scores: list[float] = []
    y: list[int] = []
    failures = 0
    for path, label in labelled:
        try:
            dets = birdnet.analyze_file(str(path), min_conf=thresh)
            score = max((conf for _s, _e, _name, conf in dets), default=0.0)
        except Exception as exc:
            failures += 1
            if failures <= 3:
                logger.warning("birdnet failed on %s: %s", path.name, exc)
            continue
        scores.append(float(score))
        y.append(int(label))

    if len(scores) < 20:
        return {
            "skip": f"only {len(scores)} usable predictions "
            f"({failures} failures); check faun-ml-cpu image"
        }
    if len(set(y)) < 2:
        return {"skip": "single class in sampled labels; cannot compute AUC"}

    scores_a = np.asarray(scores, dtype=float)
    y_a = np.asarray(y, dtype=int)
    auc = common.auc_score(y_a, scores_a)
    prec, rec = common.precision_recall(y_a, scores_a >= thresh)

    notes = (
        f"N={len(scores)} seed={seed} thr={thresh} "
        f"precision={prec:.3f} recall={rec:.3f} pos_rate={y_a.mean():.2f} "
        f"failures={failures}; CC-BY-NC-SA inventory baseline"
    )
    return {
        "model": MODEL,
        "dataset": dataset,
        "metric": METRIC,
        "value": round(auc, 4),
        "notes": notes,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from experiments.runner import build_cfg
    import os

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
