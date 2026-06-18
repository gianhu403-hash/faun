"""Эмбеддинги: единый владелец батч-экспорта эмбеддингов из аудио.

Контур: протокол :class:`Embedder` (фикс-мерный вектор на клип) + два адаптера
поверх замороженных wrapper'ов экспериментов:

* :class:`PerchEmbedder` — Perch (``experiments.wrappers.perch``): downmix mono,
  resample до 32 кГц (``soxr``), pad/truncate до окна 160000 сэмплов (5 с),
  отдаёт эмбеддинг [1280];
* :class:`YamnetEmbedder` — YAMNet (``experiments.wrappers.yamnet_probe``):
  downmix mono, resample до 16 кГц, пуленный вектор concat(mean, max) = [2048].

TensorFlow обоими адаптерами импортируется **лениво и только внутри wrapper'ов**
(`perch.embed` / `yamnet_probe.embed_waveform`), поэтому реально гонять Perch/
YAMNet можно только на кластере. Препроцессинг (resample + pad/truncate +
downmix) живёт здесь и тестируется TF-free через монкипатч wrapper-функций.

:func:`embed_batch` стекает по-клиповые эмбеддинги, :class:`EmbeddingCache`
персистит их в ``.npz`` (с опциональными id/метками детекций).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

try:
    import soxr

    _HAS_SOXR = True
except ImportError:  # pragma: no cover - soxr в requirements-pipeline.txt
    _HAS_SOXR = False

__all__ = [
    "Embedder",
    "PerchEmbedder",
    "YamnetEmbedder",
    "embed_batch",
    "EmbeddingCache",
]


# ---------------------------------------------------------------------------
# Протокол (замороженный публичный контракт)
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Контракт эмбеддера: один клип -> один фикс-мерный вектор."""

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Вернуть эмбеддинг [DIM] для ``waveform`` на частоте ``sr`` Гц."""
        ...


# ---------------------------------------------------------------------------
# Препроцессинг (общий для адаптеров, тестируется TF-free)
# ---------------------------------------------------------------------------


def _downmix(waveform: np.ndarray) -> np.ndarray:
    """Stereo/multichannel -> mono float32 через среднее по каналам."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"ожидался 1-D или 2-D сигнал, получен ndim={waveform.ndim}")
    # soundfile отдаёт (frames, channels); принимаем и (channels, frames).
    channel_axis = 1 if waveform.shape[1] <= waveform.shape[0] else 0
    return waveform.mean(axis=channel_axis).astype(np.float32)


def _resample(mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
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


def _fit_window(mono: np.ndarray, win_samples: int) -> np.ndarray:
    """Pad нулями справа или truncate до ровно ``win_samples`` сэмплов."""
    if len(mono) == win_samples:
        return mono.astype(np.float32, copy=False)
    if len(mono) > win_samples:
        return mono[:win_samples].astype(np.float32, copy=False)
    out = np.zeros(win_samples, dtype=np.float32)
    out[: len(mono)] = mono
    return out


# ---------------------------------------------------------------------------
# Адаптеры (ленивый TF внутри wrapper'ов)
# ---------------------------------------------------------------------------


class PerchEmbedder:
    """Эмбеддер Perch: downmix -> 32 кГц -> окно 160000 -> вектор [1280].

    Тяжёлый TF тянется лениво внутри ``experiments.wrappers.perch.embed``,
    так что реально работает только на кластере. Препроцессинг — здесь.
    """

    DIM = 1280

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        # Импорт по месту: модуль лёгкий, но вызываем через атрибут модуля,
        # чтобы монкипатч `perch.embed` в тестах перехватывал реальный путь.
        import experiments.wrappers.perch as perch

        mono = _downmix(waveform)
        resampled = _resample(mono, sr, perch.SR)
        window = _fit_window(resampled, perch.WIN_SAMPLES)
        embeddings, _logits = perch.embed(window[np.newaxis, :])
        return np.asarray(embeddings[0], dtype=np.float32)


class YamnetEmbedder:
    """Эмбеддер YAMNet: downmix -> 16 кГц -> concat(mean, max) = [2048].

    Повторяет пуллинг продового классификатора v7. TF — лениво внутри
    ``experiments.wrappers.yamnet_probe.embed_waveform``.
    """

    DIM = 2048

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        import experiments.wrappers.yamnet_probe as yamnet_probe

        mono = _downmix(waveform)
        resampled = _resample(mono, sr, yamnet_probe.SR)
        frames = np.asarray(yamnet_probe.embed_waveform(resampled), dtype=np.float32)
        if frames.shape[0] == 0:
            return np.zeros(self.DIM, dtype=np.float32)
        return np.concatenate([frames.mean(axis=0), frames.max(axis=0)]).astype(
            np.float32
        )


# ---------------------------------------------------------------------------
# Батч + кэш
# ---------------------------------------------------------------------------


def embed_batch(
    clips: Iterable[tuple[np.ndarray, int]], embedder: Embedder
) -> np.ndarray:
    """Стекнуть по-клиповые эмбеддинги в матрицу [N, DIM].

    ``clips`` — последовательность пар ``(waveform, sr)``. Пустой вход даёт
    пустую 2-D матрицу формы ``(0, DIM)``, где DIM берётся из ``embedder.DIM``,
    а если его нет — ``(0, 0)`` (без падения).
    """
    rows = [
        np.asarray(embedder.embed(waveform, sr), dtype=np.float32)
        for waveform, sr in clips
    ]
    if rows:
        return np.stack(rows).astype(np.float32)
    return np.zeros((0, _embedder_dim(embedder)), dtype=np.float32)


def _embedder_dim(embedder: object) -> int:
    """Размерность эмбеддера: класс-атрибут ``DIM`` или инстанс-атрибут ``dim``.

    Нужно для формы пустого батча; ``0`` если размерность недоступна.
    """
    return int(getattr(embedder, "DIM", None) or getattr(embedder, "dim", 0))


@dataclass
class EmbeddingCache:
    """Эмбеддинги + опциональные id/метки детекций с npz-персистентностью.

    Хранит матрицу ``embeddings`` [N, DIM] и, опционально, выровненные по строкам
    списки ``ids`` (например, ``detection_id``) и ``labels``. На диске — один
    ``.npz``; отсутствующие ids/labels не записываются и при загрузке дают None.
    """

    embeddings: np.ndarray
    ids: list[str] | None = None
    labels: list[str] | None = None

    def __len__(self) -> int:
        return int(np.asarray(self.embeddings).shape[0])

    def save(self, path: str | Path) -> Path:
        """Сохранить кэш в ``.npz`` (атомарно: tmp + rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "embeddings": np.asarray(self.embeddings, dtype=np.float32)
        }
        if self.ids is not None:
            payload["ids"] = np.asarray(self.ids, dtype=object)
        if self.labels is not None:
            payload["labels"] = np.asarray(self.labels, dtype=object)
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez(tmp, **payload)
        # np.savez добавляет .npz к имени без расширения; нормализуем.
        produced = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
        produced.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingCache":
        """Загрузить кэш из ``.npz``, восстановив ids/labels как list[str].

        Битый/усечённый ``.npz`` или отсутствие ключа ``embeddings`` дают явный
        ``ValueError`` с путём (а не невнятный ``BadZipFile``/``KeyError``) —
        например при прерванном ``batch_label --emb-out``.
        """
        path = Path(path)
        try:
            # allow_pickle: .npz — операторский локальный файл, записанный нашим
            # же save() (object-массивы ids/labels), НЕ сетевой ввод.
            with np.load(path, allow_pickle=True) as data:
                if "embeddings" not in data.files:
                    raise KeyError("missing 'embeddings'")
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                ids = (
                    [str(x) for x in data["ids"].tolist()]
                    if "ids" in data.files
                    else None
                )
                labels = (
                    [str(x) for x in data["labels"].tolist()]
                    if "labels" in data.files
                    else None
                )
        except Exception as exc:  # noqa: BLE001 — любой сбой чтения = битый файл
            # np.load на мусоре кидает разнотипное (BadZipFile/UnpicklingError/
            # ValueError/EOFError) — всё это означает «кэш повреждён». Цепляем
            # оригинал через ``from exc`` для отладки.
            raise ValueError(f"corrupt embedding cache: {path}") from exc
        return cls(embeddings=embeddings, ids=ids, labels=labels)
