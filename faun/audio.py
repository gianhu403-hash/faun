"""Аудио-препроцессинг: единый владелец downmix / resample / fit_window (ADR-0002).

Это канонический дом для трёх примитивов подготовки сигнала перед любым
эмбеддером/классификатором/датасетом:

* :func:`downmix` — stereo/multichannel -> mono float32 (среднее по каналам,
  с эвристикой ``(frames, channels)`` vs ``(channels, frames)``);
* :func:`resample` — ресемпл mono до целевой частоты (``soxr`` если доступен,
  иначе линейная интерполяция; ``ValueError`` при ``sr <= 0``);
* :func:`fit_window` — pad нулями справа либо truncate до ровно ``win_samples``.

Раньше эти функции дублировались в ``faun.embeddings`` и
``faun.segmentation``; теперь обе точки делегируют сюда, чтобы семантика
препроцессинга жила в одном месте. ``faun.embeddings`` дополнительно ре-экспортит
их под замороженными именами ``_downmix`` / ``_resample`` / ``_fit_window``
(тем же объектом), от которых зависит ``faun.training.dataset``.

soxr импортируется на уровне модуля (он в ``requirements-pipeline.txt``); TF/torch
здесь не используются вовсе — препроцессинг тестируется TF/torch-free.
"""

from __future__ import annotations

import numpy as np

try:
    import soxr

    _HAS_SOXR = True
except ImportError:  # pragma: no cover - soxr в requirements-pipeline.txt
    _HAS_SOXR = False

__all__ = ["downmix", "resample", "fit_window"]


def downmix(waveform: np.ndarray) -> np.ndarray:
    """Stereo/multichannel -> mono float32 через среднее по каналам."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"ожидался 1-D или 2-D сигнал, получен ndim={waveform.ndim}")
    # soundfile отдаёт (frames, channels); принимаем и (channels, frames).
    channel_axis = 1 if waveform.shape[1] <= waveform.shape[0] else 0
    return waveform.mean(axis=channel_axis).astype(np.float32)


def resample(mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample mono-сигнала до ``target_sr`` (soxr; иначе линейный фолбэк)."""
    if sr == target_sr:
        return mono.astype(np.float32, copy=False)
    if sr <= 0:
        raise ValueError(f"частота должна быть положительной, получено {sr}")
    if _HAS_SOXR:
        return soxr.resample(mono, sr, target_sr).astype(np.float32)
    # Фолбэк без soxr: линейная интерполяция (для тестов без soxr).
    n_out = int(round(len(mono) * target_sr / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_out = np.linspace(0.0, len(mono) - 1, n_out)
    return np.interp(x_out, np.arange(len(mono)), mono).astype(np.float32)


def fit_window(mono: np.ndarray, win_samples: int) -> np.ndarray:
    """Pad нулями справа или truncate до ровно ``win_samples`` сэмплов."""
    if len(mono) == win_samples:
        return mono.astype(np.float32, copy=False)
    if len(mono) > win_samples:
        return mono[:win_samples].astype(np.float32, copy=False)
    out = np.zeros(win_samples, dtype=np.float32)
    out[: len(mono)] = mono
    return out
