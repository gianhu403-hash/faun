"""Perch 1 (Google bird-vocalization-classifier) через TFHub/Kaggle. CPU-only.

Вход: окна 5 с @ 32 кГц (160000 сэмплов). embed() -> эмбеддинги [N, 1280]
и logits, если модель их отдаёт.
"""

from __future__ import annotations

import os

import numpy as np

SR = 32_000
WIN_SAMPLES = 5 * SR  # 160000

DEFAULT_URL = os.environ.get(
    "PERCH_TFHUB_URL",
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "TensorFlow2/bird-vocalization-classifier/4",
)

_model = None


def load_model(url: str = DEFAULT_URL):
    global _model
    if _model is None:
        import tensorflow_hub as hub

        _model = hub.load(url)
    return _model


def _infer(model, batch):
    """Унифицирует выход разных версий Perch -> (embeddings, logits|None)."""
    fn = getattr(model, "infer_tf", None) or model
    out = fn(batch)
    if isinstance(out, dict):
        emb = out.get("embedding")
        logits = out.get("label")
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        logits, emb = out
    else:
        emb, logits = out, None
    to_np = lambda t: None if t is None else np.asarray(t)
    return to_np(emb), to_np(logits)


def embed(windows_32k: np.ndarray, batch_size: int = 8):
    """windows_32k: [N, 160000] float32 @32kHz -> (embeddings [N, D], logits|None)."""
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
