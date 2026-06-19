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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

# Препроцессинг (downmix/resample/fit_window) живёт в faun.audio (ADR-0002).
# Ре-экспортим под замороженными именами _downmix/_resample/_fit_window тем же
# объектом — faun.training.dataset импортирует их отсюда (frozen контракт).
from faun.audio import downmix as _downmix
from faun.audio import fit_window as _fit_window
from faun.audio import resample as _resample

logger = logging.getLogger(__name__)

__all__ = [
    "Embedder",
    "PerchEmbedder",
    "Perch2Embedder",
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


class Perch2Embedder:
    """Эмбеддер Perch 2: downmix -> 32 кГц -> окно 160000 -> вектор [1536].

    Тяжёлый TF тянется лениво внутри ``experiments.wrappers.perch_v2.embed``,
    так что реально работает только на кластере. Препроцессинг — здесь.
    Размерность 1536 (Apache-2.0 Perch 2), НЕ 1280 как у Perch 1.
    """

    DIM = 1536

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        import experiments.wrappers.perch_v2 as perch_v2

        mono = _downmix(waveform)
        resampled = _resample(mono, sr, perch_v2.SR)
        window = _fit_window(resampled, perch_v2.WIN_SAMPLES)
        embeddings, _logits = perch_v2.embed(window[np.newaxis, :])
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

    Нужно для формы пустого батча; ``0`` если размерность недоступна (тогда
    пишем warning — пустой батч получит форму ``(0, 0)``, что может скрыть
    рассогласование размерностей выше по стеку).
    """
    dim = int(getattr(embedder, "DIM", None) or getattr(embedder, "dim", 0))
    if dim == 0:
        logger.warning(
            "embedder %r exposes neither DIM nor dim; empty batch shape is (0, 0)",
            type(embedder).__name__,
        )
    return dim


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
