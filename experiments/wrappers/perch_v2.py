"""Perch 2 (Google bird-vocalization-classifier) через Kaggle SavedModel. CPU/GPU.

Вход: окна 5 с @ 32 кГц (160000 сэмплов). ``embed()`` -> эмбеддинги [N, 1536]
и logits (выход ``label``). В отличие от Perch 1 (``hub.load().infer_tf``),
Perch 2 вызывается через serving-сигнатуру SavedModel::

    model = tf.saved_model.load(path)
    out = model.signatures["serving_default"](inputs=data)  # data: (N, 160000) float32
    # out -> {"embedding"(N,1536), "spatial_embedding"(N,5,3,1536), "label", "spectrogram"}

TF и kagglehub тянутся **лениво** внутри ``load_model`` — импорт модуля их не
подтягивает. Источник модели: ``PERCH_V2_MODEL_PATH`` (локальный SavedModel)
либо ``kagglehub.model_download(<handle>)`` (нужны Kaggle-креды; модель gated).
"""

from __future__ import annotations

import os

import numpy as np

SR = 32_000
WIN_SAMPLES = 5 * SR  # 160000
DIM = 1536

HANDLE_GPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
HANDLE_CPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"

_model = None


def load_model(model_path: str | None = None, cpu: bool = False):
    """Лениво загрузить Perch 2 SavedModel (кэш в модульной глобали).

    Источник: ``model_path`` -> ``PERCH_V2_MODEL_PATH`` -> kagglehub-загрузка.
    """
    global _model
    if _model is not None:
        return _model
    import tensorflow as tf

    path = model_path or os.environ.get("PERCH_V2_MODEL_PATH")
    if not path:
        import kagglehub

        path = kagglehub.model_download(HANDLE_CPU if cpu else HANDLE_GPU)
    _model = tf.saved_model.load(path)
    return _model


def _infer(model, batch):
    """Прогнать serving-сигнатуру Perch 2 -> (embeddings [N,1536], logits|None)."""
    serving = model.signatures["serving_default"]
    out = serving(inputs=batch)
    emb = out["embedding"] if "embedding" in out else None
    logits = out["label"] if "label" in out else out.get("logits")
    to_np = lambda t: (
        None if t is None else np.asarray(t.numpy() if hasattr(t, "numpy") else t)
    )
    return to_np(emb), to_np(logits)


def embed(windows_32k: np.ndarray, batch_size: int = 8):
    """windows_32k: [N, 160000] float32 @32kHz -> (embeddings [N, 1536], logits|None)."""
    windows_32k = np.asarray(windows_32k, dtype=np.float32)
    if windows_32k.ndim != 2 or windows_32k.shape[1] != WIN_SAMPLES:
        raise ValueError(
            f"expected [N, {WIN_SAMPLES}] (5s @ {SR}Hz), got {windows_32k.shape}"
        )
    model = load_model()
    embs, logits_parts = [], []
    for i in range(0, len(windows_32k), batch_size):
        emb, logits = _infer(model, windows_32k[i : i + batch_size])
        embs.append(emb)
        if logits is not None:
            logits_parts.append(logits)
    embeddings = np.concatenate(embs, axis=0)
    all_logits = np.concatenate(logits_parts, axis=0) if logits_parts else None
    return embeddings, all_logits
