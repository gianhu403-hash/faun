"""ESC-50 bird/no-bird probe — Perch + YAMNet эмбеддинги (CPU/TF).

Standalone-эксперимент (НЕ ломает exp_e2/e5). Задача: "chirping_birds vs всё
остальное" на ESC-50. 40 позитивов + сбалансированный сид-сэмпл негативов.
Эмбеддим каждый клип, гоняем StandardScaler+LogisticRegression с 5-fold
StratifiedCV, репортим mean ROC-AUC и дописываем ДВЕ строки в results.csv.

Perch:  ресэмпл 32k, окна 5с (160000), mean-pool окон -> вектор клипа.
YAMNet: ресэмпл 16k, mean+max pool фреймов -> 2048 (как прод v7).

Запуск (docker):
  docker run --rm -v /home/oleg/faun-data:/data:Z \
    -v /home/oleg/faun-data/code:/code:Z -e PYTHONPATH=/code \
    -e HF_HOME=/data/hf_cache -w /code faun-ml-cpu \
    python -m experiments.exp_esc50_probe --data-root /data
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

from experiments import common

logger = logging.getLogger("faun.experiments.esc50_probe")

DATASET = "esc50_birds"
METRIC = "roc_auc"
POS_CATEGORY = "chirping_birds"
SEED = 1337
N_NEG = 200            # ~200 негативов на 40 позитивов (сбалансировано-ish)
PERCH_SR = 32000
PERCH_WIN_S = 5.0
YAMNET_SR = 16000


def _esc50_paths(data_root: Path) -> tuple[Path, Path]:
    """-> (meta_csv, audio_dir). Поддерживает layout с ESC-50-master/."""
    base = data_root / "datasets" / "esc50"
    for prefix in (base / "ESC-50-master", base):
        meta = prefix / "meta" / "esc50.csv"
        audio = prefix / "audio"
        if meta.exists() and audio.is_dir():
            return meta, audio
    raise FileNotFoundError(f"ESC-50 meta/audio not found under {base}")


def _build_sample(meta_csv: Path, audio_dir: Path):
    """Читает meta CSV, собирает [(path, label)] — все позитивы + сид-сэмпл негативов."""
    import random

    pos, neg = [], []
    with open(meta_csv, newline="") as f:
        for row in csv.DictReader(f):
            p = audio_dir / row["filename"]
            if not p.exists():
                continue
            if row["category"] == POS_CATEGORY:
                pos.append(p)
            else:
                neg.append(p)
    pos = sorted(pos)
    neg = sorted(neg)
    rng = random.Random(SEED)
    if len(neg) > N_NEG:
        neg = sorted(rng.sample(neg, N_NEG))
    labelled = [(p, 1) for p in pos] + [(p, 0) for p in neg]
    return labelled, len(pos), len(neg)


# ------------------------------------------------------------------ embedders


def _perch_vectors(labelled):
    """mean-pool Perch-эмбеддингов 5с-окон @32k -> (X, y, failures, dim)."""
    from experiments.wrappers import perch

    X, y, failures = [], [], 0
    for path, label in labelled:
        try:
            wav, sr = common.load_audio(path, target_sr=PERCH_SR, mono=True)
            wins = common.windows(wav, sr, win_s=PERCH_WIN_S)  # [n, 160000]
            embs, _ = perch.embed(wins)
            embs = np.asarray(embs)
            if embs.size == 0:
                raise ValueError("empty embeddings")
            X.append(embs.mean(axis=0).ravel())
            y.append(int(label))
        except Exception as exc:
            failures += 1
            if failures <= 3:
                logger.warning("perch embed failed on %s: %s", path.name, exc)
    dim = int(np.asarray(X[0]).size) if X else 0
    return X, y, failures, dim


def _yamnet_vectors(labelled):
    """mean+max-pool YAMNet-фреймов @16k -> (X, y, failures, dim)."""
    from experiments.wrappers import yamnet_probe

    X, y, failures = [], [], 0
    for path, label in labelled:
        try:
            vec = yamnet_probe.embed_file(path)  # ресэмпл 16k внутри
            vec = np.asarray(vec)
            if vec.size == 0 or not np.any(np.isfinite(vec)):
                raise ValueError("empty/non-finite embedding")
            X.append(vec.ravel())
            y.append(int(label))
        except Exception as exc:
            failures += 1
            if failures <= 3:
                logger.warning("yamnet embed failed on %s: %s", path.name, exc)
    dim = int(np.asarray(X[0]).size) if X else 0
    return X, y, failures, dim


# ------------------------------------------------------------------- CV probe


def _cv_auc(X, y, seed: int, folds: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = max(2, min(folds, int(np.bincount(y).min())))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")))


# ----------------------------------------------------------------------- main


def _run_one(model_name, vectors_fn, labelled, results_csv):
    t0 = time.time()
    try:
        X, y, failures, dim = vectors_fn(labelled)
    except Exception as exc:
        runtime = round(time.time() - t0, 1)
        note = f"{type(exc).__name__}: {exc}"
        logger.error("%s FAILED: %s", model_name, note)
        common.append_result(
            results_csv, model_name, DATASET, "status", "error",
            runtime, "", note,
        )
        return {"model": model_name, "error": note}

    runtime = round(time.time() - t0, 1)
    if len(X) < 20 or len(set(y)) < 2:
        note = f"insufficient embeddings: N={len(X)} failures={failures}"
        logger.error("%s skip: %s", model_name, note)
        common.append_result(
            results_csv, model_name, DATASET, "status", "skip",
            runtime, "", note,
        )
        return {"model": model_name, "skip": note}

    Xa, ya = np.stack(X), np.asarray(y)
    auc = round(_cv_auc(Xa, ya, SEED), 4)
    notes = (
        f"N={len(X)} dim={dim} 5-fold-CV seed={SEED} "
        f"pos={int(ya.sum())} neg={int((ya == 0).sum())} "
        f"pos_rate={ya.mean():.2f} failures={failures} "
        f"task={POS_CATEGORY}_vs_rest; frozen emb + StdScaler+LogReg"
    )
    common.append_result(
        results_csv, model_name, DATASET, METRIC, auc, runtime, "", notes,
    )
    logger.info("%s: %s=%.4f (%ss, %s)", model_name, METRIC, auc, runtime, notes)
    return {"model": model_name, "metric": METRIC, "value": auc, "notes": notes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_csv = data_root / "results" / "results.csv"

    meta_csv, audio_dir = _esc50_paths(data_root)
    labelled, n_pos, n_neg = _build_sample(meta_csv, audio_dir)
    logger.info("ESC-50 sample: %d pos + %d neg = %d clips",
                n_pos, n_neg, len(labelled))

    results = []
    # Perch first (TFHub download); YAMNet second so a Perch failure still yields YAMNet.
    results.append(_run_one("perch_v1_esc50", _perch_vectors, labelled, results_csv))
    results.append(_run_one("yamnet_base_esc50", _yamnet_vectors, labelled, results_csv))

    print("=== ESC-50 probe results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
