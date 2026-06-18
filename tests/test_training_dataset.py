"""Тесты ``iNatTorchDataset`` — TF/torch-FREE на MINI-фикстуре iNatSounds.

ABSOLUTE RULE: НЕТ ``import torch`` на уровне модуля. Препроцессинг датасета
(downmix -> resample -> фикс-окно) гоняется на numpy-фолбэке
``faun.embeddings._resample`` (torch-free). Если torch присутствует локально,
``__getitem__`` оборачивает сигнал в ``torch.Tensor`` — это проверяется отдельно
и устойчиво к его отсутствию.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faun.training.dataset import DEFAULT_SR, iNatTorchDataset, make_loaders

FIXTURE = Path(__file__).parent / "fixtures" / "inatsounds_mini"


def _to_numpy(waveform):
    """Привести элемент к numpy независимо от того, Tensor он или ndarray."""
    if hasattr(waveform, "detach"):  # torch.Tensor
        return waveform.detach().cpu().numpy()
    return np.asarray(waveform)


def test_import_does_not_pull_torch():
    """STUB/torch-free: импорт faun.training не должен тянуть torch в sys.modules.

    Проверяем, что коллекция/импорт пакета не импортирует torch жадно (PEP-562).
    """
    import sys

    # удалим возможный след предыдущих тестов
    had_torch = "torch" in sys.modules
    import importlib

    import faun.training  # noqa: F401

    importlib.reload(faun.training)
    if not had_torch:
        assert "torch" not in sys.modules, "import faun.training не должен тянуть torch"


def test_dataset_yields_waveform_and_label_idx():
    """torch-free: каждый элемент = (waveform[win], label_idx) с верным vocab."""
    ds = iNatTorchDataset(str(FIXTURE), win_s=0.5, sr=16_000)
    vocab = ds.vocab
    # MINI: 3 вида, 5 клипов.
    assert len(ds) == 5
    assert set(vocab) == {"Erithacus_rubecula", "Parus_major", "Turdus_merula"}
    assert sorted(vocab.values()) == [0, 1, 2]

    waveform, label_idx = ds[0]
    wf = _to_numpy(waveform)
    # фикс-окно: 0.5с * 16000 = 8000 сэмплов.
    assert wf.shape == (8000,)
    assert wf.dtype == np.float32
    assert isinstance(label_idx, int)
    assert label_idx in vocab.values()


def test_label_idx_matches_species_folder():
    """torch-free: label_idx соответствует виду из имени папки через vocab."""
    ds = iNatTorchDataset(str(FIXTURE), win_s=0.5, sr=16_000)
    vocab = ds.vocab
    for i in range(len(ds)):
        _wf, label_idx = ds[i]
        rec = ds._records[i]  # noqa: SLF001 - проверяем соответствие
        assert label_idx == vocab[rec.species]


def test_resample_to_32k_and_window_applied():
    """torch-free: ресемпл 16k->32k + фикс-окно 10с => 320000 сэмплов."""
    ds = iNatTorchDataset(str(FIXTURE), sr=DEFAULT_SR, win_s=10.0)
    waveform, _ = ds[0]
    wf = _to_numpy(waveform)
    assert wf.shape == (DEFAULT_SR * 10,)  # 320000


def test_deterministic_by_record_order():
    """torch-free: два инстанса дают идентичные сигналы (детерминизм обхода)."""
    ds_a = iNatTorchDataset(str(FIXTURE), win_s=0.5, sr=16_000)
    ds_b = iNatTorchDataset(str(FIXTURE), win_s=0.5, sr=16_000)
    for i in range(len(ds_a)):
        wa, la = ds_a[i]
        wb, lb = ds_b[i]
        assert la == lb
        np.testing.assert_array_equal(_to_numpy(wa), _to_numpy(wb))


def test_records_subset_via_split():
    """torch-free: передача records=train_split ограничивает датасет этим сабсетом."""
    from faun.datasets import iNatSoundsDataset

    full = iNatSoundsDataset(str(FIXTURE))
    vocab = full.vocab()
    train, val = full.split(seed=42)

    train_ds = iNatTorchDataset(
        str(FIXTURE), vocab, records=train, win_s=0.5, sr=16_000
    )
    val_ds = iNatTorchDataset(str(FIXTURE), vocab, records=val, win_s=0.5, sr=16_000)
    assert len(train_ds) == len(train)
    assert len(val_ds) == len(val)
    assert len(train_ds) + len(val_ds) == 5
    # vocab общий для обоих сплитов.
    assert train_ds.vocab == vocab == val_ds.vocab


@pytest.mark.requires_torch
def test_getitem_returns_torch_tensor_when_torch_present():
    """TORCH-ONLY: с установленным torch __getitem__ отдаёт FloatTensor."""
    torch = pytest.importorskip("torch")
    ds = iNatTorchDataset(str(FIXTURE), win_s=0.5, sr=16_000)
    waveform, _label = ds[0]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.shape == (8000,)


@pytest.mark.requires_torch
def test_make_loaders_builds_torch_dataloaders():
    """TORCH-ONLY: make_loaders отдаёт два DataLoader со сплитом по seed."""
    pytest.importorskip("torch")
    from faun.datasets import iNatSoundsDataset

    vocab = iNatSoundsDataset(str(FIXTURE)).vocab()
    train_loader, val_loader = make_loaders(
        str(FIXTURE), vocab, seed=42, sr=16_000, win_s=0.5, batch_size=2
    )
    # один батч из train проходит без ошибок и имеет правильную форму окна.
    batch_wf, batch_labels = next(iter(train_loader))
    assert batch_wf.shape[1] == 8000
    assert batch_wf.shape[0] == batch_labels.shape[0]
