"""E2 — Perch embeddings + linear probe on the bird-detection subset (CPU/TF).

Goal
----
Measure how well frozen Perch embeddings separate bird vs no-bird with a
cheap linear head. We embed 5-second windows @32kHz of each labelled clip
(freefield1010 / warblrb10k), mean-pool windows to one vector per clip, then
run a standardized LogisticRegression with 5-fold stratified CV and report
mean ROC-AUC.

No fine-tuning — frozen embeddings + linear probe only (Perch is Apache-2.0,
so the probe is unencumbered, unlike BirdNET in E1). CPU: Perch runs on
TF/JAX without a GPU; this is the recommended classifier candidate.

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
Wrapper: experiments.wrappers.perch.embed(windows_32k [N,160000]) ->
(embeddings [N,D], logits|None). Missing wrapper/data -> graceful skip.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiments import common
from experiments.exp_e1 import _find_dataset, _find_label_csv

logger = logging.getLogger("faun.experiments.e2")

MODEL = "Perch"
METRIC = "roc_auc"
DEFAULT_N = 400
DEFAULT_SEED = 1337
WIN_S = 5.0
TARGET_SR = 32000  # Perch native rate (perch wrapper expects 160000-sample windows)


def _clip_vectors(cfg, labelled):
    """Embed each clip's windows with Perch, mean-pool -> (X, y, failures)."""
    import numpy as np

    from experiments.wrappers import perch

    X, y, failures = [], [], 0
    for path, label in labelled:
        try:
            wav, sr = common.load_audio(path, target_sr=TARGET_SR, mono=True)
            wins = common.windows(wav, sr, win_s=WIN_S)  # [n, 160000]
            embs, _logits = perch.embed(wins)
            embs = np.asarray(embs)
            if embs.size == 0:
                raise ValueError("empty embeddings")
            X.append(embs.mean(axis=0).ravel())
            y.append(int(label))
        except Exception as exc:
            failures += 1
            if failures <= 3:
                logger.warning("perch embed failed on %s: %s", Path(path).name, exc)
    return X, y, failures


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
        from experiments.wrappers import perch  # noqa: F401  (presence check)
    except Exception as exc:
        return {"skip": f"perch wrapper unavailable: {type(exc).__name__}: {exc}"}

    X, y, failures = _clip_vectors(cfg, labelled)
    if len(X) < 20:
        return {
            "skip": f"only {len(X)} embeddings ({failures} failures); "
            "check perch/Kaggle model availability"
        }
    if len(set(y)) < 2:
        return {"skip": "single class in sampled labels; cannot compute AUC"}

    auc = _cv_auc(np.stack(X), np.asarray(y), seed)
    dim = int(np.asarray(X[0]).size)
    notes = (
        f"N={len(X)} dim={dim} 5-fold-CV win={WIN_S}s seed={seed} "
        f"pos_rate={np.mean(y):.2f} failures={failures}; "
        "frozen embeddings + LogReg (Apache-2.0)"
    )
    return {
        "model": MODEL,
        "dataset": dataset,
        "metric": METRIC,
        "value": round(auc, 4),
        "notes": notes,
    }


def _cv_auc(X, y, seed: int, folds: int = 5) -> float:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = max(2, min(folds, int(np.bincount(y).min())))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from experiments.runner import build_cfg

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
