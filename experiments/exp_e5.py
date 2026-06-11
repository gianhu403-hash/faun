"""E5 — YAMNet embeddings + linear probe on the bird-detection subset.

Goal
----
Quantify the cost of staying on YAMNet vs switching to a bioacoustics-native
model (Perch, E2). We take YAMNet's *base* tfhub embeddings — the yamnet_probe
wrapper pools frame embeddings to a 2048-d clip vector (mean+max), exactly as
the prod v7 classifier does; the v7 *anthropic head* is NOT used (bird
detection is outside its 6-class chainsaw/gunshot vocabulary). Same labelled
freefield1010/warblrb10k subset and same standardized LogReg 5-fold CV as E2,
so E2 (Perch) vs E5 (YAMNet) is a like-for-like model-swap comparison on
ROC-AUC.

Rationale for the swap: YAMNet is trained on AudioSet (general audio), not
bird vocalisations; its embedding manifold is not tuned for fine species
structure, so we expect lower AUC than Perch. E5 produces the number that
justifies adopting Perch for the bird-monitoring pilot.

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
Wrapper: experiments.wrappers.yamnet_probe.embed_file(path) -> [2048].
Missing wrapper/base model/data -> graceful skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiments import common
from experiments.exp_e1 import _find_dataset, _find_label_csv
from experiments.exp_e2 import _cv_auc

logger = logging.getLogger("faun.experiments.e5")

MODEL = "YAMNet-base"
METRIC = "roc_auc"
DEFAULT_N = 400
DEFAULT_SEED = 1337


def run(cfg: dict) -> dict:
    import numpy as np

    seed = int(cfg.get("seed", DEFAULT_SEED))
    n = int(cfg.get("n_samples", DEFAULT_N))

    ds_dir = _find_dataset(cfg)
    if ds_dir is None:
        return {"skip": "no labelled bird-detection dataset under datasets/"}
    dataset = ds_dir.name

    label_csv = _find_label_csv(ds_dir)
    if label_csv is None:
        return {"skip": f"no label CSV in {dataset}"}
    try:
        labels = common.read_bird_labels(label_csv)
    except (KeyError, ValueError) as exc:
        return {"skip": f"label CSV not freefield/warblr schema: {exc}"}
    if not labels:
        return {"skip": f"empty labels in {label_csv.name}"}

    files = common.sample_files(ds_dir, n=n, seed=seed)
    labelled = [(p, labels[p.stem]) for p in files if p.stem in labels]
    if len(labelled) < 20:
        return {"skip": f"only {len(labelled)} labelled files matched"}

    try:
        from experiments.wrappers import yamnet_probe
    except Exception as exc:
        return {
            "skip": f"yamnet_probe wrapper unavailable: {type(exc).__name__}: {exc}"
        }

    X, y, failures = [], [], 0
    for path, label in labelled:
        try:
            vec = np.asarray(
                yamnet_probe.embed_file(str(path)), dtype="float32"
            ).ravel()
        except Exception as exc:
            failures += 1
            if failures <= 3:
                logger.warning("yamnet embed failed %s: %s", path.name, exc)
            continue
        if vec.size == 0:
            failures += 1
            continue
        X.append(vec)
        y.append(int(label))

    if len(X) < 20:
        return {
            "skip": f"only {len(X)} embeddings ({failures} failures); "
            "check tfhub YAMNet base availability"
        }
    if len(set(y)) < 2:
        return {"skip": "single class in sampled labels; cannot compute AUC"}

    auc = _cv_auc(np.stack(X), np.asarray(y), seed)
    dim = int(np.asarray(X[0]).size)
    notes = (
        f"N={len(X)} dim={dim} 5-fold-CV seed={seed} "
        f"pos_rate={np.mean(y):.2f} failures={failures}; "
        "base AudioSet embeddings (no v7 head). Compare vs E2(Perch) "
        "to justify model swap"
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
    import os
    from experiments.runner import build_cfg

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
