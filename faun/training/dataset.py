"""Torch-датасет iNatSounds для fine-tune трансформера (raw-waveform).

Переиспользует:

* :class:`faun.datasets.iNatSoundsDataset` — разбор дерева ``root/<species>/<clip>``,
  словарь видов и стратифицированный сплит (чистый stdlib+numpy, без torch);
* :func:`faun.embeddings._downmix` / ``_resample`` / ``_fit_window`` — единый
  препроцессинг (downmix mono -> resample -> фикс-окно), чтобы не дублировать
  логику ресемплинга.

PaSST нативно 32 кГц и сам гонит mel-фронтенд из raw-waveform, поэтому датасет
отдаёт именно сырой сигнал (resample до 32k, фикс-окно 10 с), без предрасчёта
мелов. torch импортируется **лениво** внутри ``__getitem__`` — сам сигнал
готовится на numpy, в ``torch.Tensor`` оборачивается только если torch есть.
Поэтому ``iNatTorchDataset`` тестируется TF/torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from faun.datasets import iNatSoundsDataset
from faun.embeddings import _downmix, _fit_window, _resample

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

# Дефолты PaSST: 32 кГц, окно 10 с.
DEFAULT_SR = 32_000
DEFAULT_WIN_S = 10.0


def _torch_dataset_base():
    """Лениво вернуть ``torch.utils.data.Dataset`` либо ``object`` (torch отсутствует).

    Базовый класс выбирается во время **инстанцирования**, а не импорта модуля,
    чтобы импорт ``faun.training.dataset`` не требовал torch. Когда torch есть —
    наследуемся от него (нужно для DataLoader); иначе работаем как обычный
    sequence-like объект (``__len__`` + ``__getitem__``), достаточный для тестов.
    """
    try:
        import torch.utils.data as tud
    except Exception:  # noqa: BLE001 - torch отсутствует (CI без torch)
        return object
    return tud.Dataset


class iNatTorchDataset:
    """``(waveform, label_idx)`` поверх дерева iNatSounds, готовый под трансформер.

    Каждый ``__getitem__`` читает аудиофайл, делает downmix -> resample до ``sr`` ->
    фикс-окно ``win_s`` секунд и возвращает ``(waveform, label_idx)``:

    * ``waveform`` — ``torch.FloatTensor`` формы ``[win_samples]`` если torch есть,
      иначе ``np.ndarray float32`` (torch-free путь для тестов);
    * ``label_idx`` — целочисленный id вида из ``vocab``.

    Детерминированность по ``seed`` обеспечивается тем, что порядок записей —
    из ``iNatSoundsDataset.manifest()`` (отсортированный обход дерева), а сам
    seed используется только в ``make_loaders`` (сплит/шаффл лоадера).

    Аргумент ``records`` позволяет передать уже посплитованный
    ``list[iNatRecord]`` (train либо val); если ``None`` — берётся весь manifest.
    """

    def __init__(
        self,
        root: str,
        vocab: "Mapping[str, int]" | None = None,
        *,
        records: "Sequence[Any]" | None = None,
        sr: int = DEFAULT_SR,
        win_s: float = DEFAULT_WIN_S,
    ) -> None:
        self._ds = iNatSoundsDataset(root)
        self._vocab: dict[str, int] = (
            dict(vocab) if vocab is not None else self._ds.vocab()
        )
        self._records: list[Any] = (
            list(records) if records is not None else self._ds.manifest()
        )
        self.sr = int(sr)
        self.win_s = float(win_s)
        self.win_samples = int(round(self.sr * self.win_s))

        # На случай DataLoader — динамически примешиваем torch.Dataset как базу.
        base = _torch_dataset_base()
        if base is not object and not isinstance(self, base):
            # Меняем класс инстанса на подкласс, наследующий torch.Dataset.
            # Делается лениво, чтобы импорт модуля оставался torch-free.
            self.__class__ = _torch_subclass(type(self), base)

    @property
    def vocab(self) -> dict[str, int]:
        """``species -> int id`` словарь, использованный для меток."""
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._records)

    def _load_waveform(self, path: str) -> np.ndarray:
        """Прочитать аудио и подготовить numpy float32 [win_samples].

        soundfile импортируется лениво. Препроцессинг — общие хелперы
        ``faun.embeddings`` (без дублирования): downmix -> resample -> окно.
        """
        import soundfile as sf

        data, file_sr = sf.read(path, dtype="float32", always_2d=False)
        mono = _downmix(np.asarray(data, dtype=np.float32))
        resampled = _resample(mono, int(file_sr), self.sr)
        return _fit_window(resampled, self.win_samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        rec = self._records[index]
        label_idx = self._vocab[rec.species]
        waveform = self._load_waveform(rec.path)

        # torch — лениво: оборачиваем в Tensor только если он установлен.
        try:
            import torch
        except Exception:  # noqa: BLE001 - torch отсутствует => numpy путь
            return waveform, int(label_idx)
        return torch.from_numpy(np.ascontiguousarray(waveform)), int(label_idx)


# Кэш сгенерированных torch-подклассов, чтобы isinstance/pickle были стабильны.
_TORCH_SUBCLASS_CACHE: dict[type, type] = {}


def _torch_subclass(cls: type, torch_base: type) -> type:
    """Создать (один раз) подкласс ``cls``, наследующий ``torch.Dataset``."""
    cached = _TORCH_SUBCLASS_CACHE.get(cls)
    if cached is not None:
        return cached
    new_cls = type(cls.__name__, (cls, torch_base), {})
    _TORCH_SUBCLASS_CACHE[cls] = new_cls
    return new_cls


def make_loaders(
    root: str,
    vocab: "Mapping[str, int]",
    *,
    seed: int,
    sr: int = DEFAULT_SR,
    win_s: float = DEFAULT_WIN_S,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[Any, Any]:
    """Построить ``(train_loader, val_loader)`` поверх iNatSounds (FROZEN сигнатура).

    Сплит — детерминированный ``iNatSoundsDataset.split(seed)`` (стратификация по
    видам). torch и его ``DataLoader`` импортируются **лениво** здесь; на машине
    без torch функция явно поднимает ``RuntimeError`` (control-flow тесты лупа
    инжектят свои лоадеры через ``_loaders`` и сюда не заходят).
    """
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "make_loaders requires PyTorch (install requirements-train.txt; "
            "real fine-tune runs on cluster-alex faun-ml-torch). "
            "Control-flow tests inject loaders via finetune(_loaders=...)."
        ) from exc

    ds = iNatSoundsDataset(root)
    train_records, val_records = ds.split(seed)

    train_ds = iNatTorchDataset(root, vocab, records=train_records, sr=sr, win_s=win_s)
    val_ds = iNatTorchDataset(root, vocab, records=val_records, sr=sr, win_s=win_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
