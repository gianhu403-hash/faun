"""YAMNet-эмбеддинги (faun.ml.yamnet) + linear probe (sklearn LogisticRegression).

CPU-only (TF). Фичи на файл: mean+max pooling фреймовых эмбеддингов -> 2048,
как в продовом классификаторе v7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16_000


def embed_waveform(x_16k: np.ndarray) -> np.ndarray:
    """Фреймовые эмбеддинги YAMNet [n_frames, 1024] для mono 16 кГц сигнала."""
    from faun.ml.yamnet import _load_models

    yamnet, _ = _load_models()
    x_16k = np.asarray(x_16k, dtype=np.float32)
    peak = np.max(np.abs(x_16k))
    if peak > 1e-6:
        x_16k = x_16k / peak
    _, embeddings, _ = yamnet(x_16k)
    return embeddings.numpy()


def embed_file(path: str | Path) -> np.ndarray:
    """Пуленый вектор файла [2048] = concat(mean, max) по фреймам."""
    from experiments.common import load_audio

    x, _ = load_audio(path, target_sr=SR, mono=True)
    emb = embed_waveform(x)
    if emb.shape[0] == 0:
        return np.zeros(2048, dtype=np.float32)
    return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])


def train_probe(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """LogisticRegression probe на эмбеддингах."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def eval_probe(clf, X: np.ndarray, y: np.ndarray) -> dict:
    """-> {'auc': ..., 'precision': ..., 'recall': ...} (метрики из common)."""
    from experiments.common import auc_score, precision_recall

    scores = clf.predict_proba(X)[:, 1]
    precision, recall = precision_recall(y, scores >= 0.5)
    return {"auc": auc_score(y, scores), "precision": precision, "recall": recall}
