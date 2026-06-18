"""Тесты загрузчика iNatSounds — без TensorFlow, без скачивания датасета.

Покрывают: разбор дерева ``root/<species>/<audiofile>`` на MINI-фикстуре,
детерминированный отсортированный ``vocab``, полноту ``manifest`` (все файлы),
стратифицированный воспроизводимый ``split(seed)`` и режимы отказа
(нет root, пустая папка вида, класс из одного примера).

Аудио — крошечные тишины через soundfile (как в test_retraining.py); ML тут нет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from faun.datasets import iNatSoundsDataset

SR = 16_000


# ---------------------------------------------------------------------------
# Построение MINI-фикстуры
# ---------------------------------------------------------------------------


def _write_clip(path: Path, n: int = SR // 8) -> None:
    """Записать крошечный тихий wav (содержимое не важно — парсим только дерево)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(n, dtype=np.float32), SR)


def _build_tree(root: Path, layout: dict[str, int]) -> Path:
    """Создать дерево ``root/<species>/<i>.wav`` по карте ``{species: n_files}``."""
    for species, count in layout.items():
        species_dir = root / species
        species_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            _write_clip(species_dir / f"{species}_{i}.wav")
    return root


@pytest.fixture
def mini_root(tmp_path: Path) -> Path:
    """MINI iNatSounds: 3 вида, разное число примеров (стратификация заметна)."""
    return _build_tree(
        tmp_path / "inat",
        {
            "Erithacus_rubecula": 6,
            "Parus_major": 4,
            "Turdus_merula": 2,
        },
    )


# ---------------------------------------------------------------------------
# vocab — детерминированный и отсортированный
# ---------------------------------------------------------------------------


def test_vocab_is_sorted_and_contiguous(mini_root: Path) -> None:
    ds = iNatSoundsDataset(mini_root)
    vocab = ds.vocab()

    assert list(vocab.keys()) == [
        "Erithacus_rubecula",
        "Parus_major",
        "Turdus_merula",
    ]
    # id — непрерывные 0..k-1 в отсортированном порядке имён.
    assert list(vocab.values()) == [0, 1, 2]


def test_vocab_deterministic_across_instances(mini_root: Path) -> None:
    assert iNatSoundsDataset(mini_root).vocab() == iNatSoundsDataset(mini_root).vocab()


# ---------------------------------------------------------------------------
# manifest — покрывает все файлы
# ---------------------------------------------------------------------------


def test_manifest_covers_every_file(mini_root: Path) -> None:
    ds = iNatSoundsDataset(mini_root)
    records = ds.manifest()

    assert len(records) == 12  # 6 + 4 + 2
    species_counts: dict[str, int] = {}
    for rec in records:
        # запись несёт path и species (dataclass или mapping).
        path = rec.path if hasattr(rec, "path") else rec["path"]
        species = rec.species if hasattr(rec, "species") else rec["species"]
        assert Path(path).exists()
        assert Path(path).suffix == ".wav"
        species_counts[species] = species_counts.get(species, 0) + 1

    assert species_counts == {
        "Erithacus_rubecula": 6,
        "Parus_major": 4,
        "Turdus_merula": 2,
    }


# ---------------------------------------------------------------------------
# split — стратифицированный и воспроизводимый
# ---------------------------------------------------------------------------


def _species_of(rec) -> str:
    return rec.species if hasattr(rec, "species") else rec["species"]


def test_split_is_reproducible_by_seed(mini_root: Path) -> None:
    ds = iNatSoundsDataset(mini_root)
    train_a, val_a = ds.split(seed=42)
    train_b, val_b = ds.split(seed=42)

    def paths(recs):
        return [r.path if hasattr(r, "path") else r["path"] for r in recs]

    assert paths(train_a) == paths(train_b)
    assert paths(val_a) == paths(val_b)


def test_split_differs_by_seed(mini_root: Path) -> None:
    ds = iNatSoundsDataset(mini_root)

    def paths(recs):
        return [r.path if hasattr(r, "path") else r["path"] for r in recs]

    _, val_1 = ds.split(seed=1)
    _, val_2 = ds.split(seed=2)
    # Разные seed дают (в общем случае) разные val-наборы.
    assert paths(val_1) != paths(val_2)


def test_split_is_stratified_and_complete(mini_root: Path) -> None:
    ds = iNatSoundsDataset(mini_root)
    train, val = ds.split(seed=7)

    # Объединение train+val = весь manifest, без потерь и дублей.
    assert len(train) + len(val) == 12
    train_paths = {r.path if hasattr(r, "path") else r["path"] for r in train}
    val_paths = {r.path if hasattr(r, "path") else r["path"] for r in val}
    assert train_paths.isdisjoint(val_paths)

    # Каждый вид с >=2 примерами представлен в обоих сплитах.
    for split in (train, val):
        present = {_species_of(r) for r in split}
        assert "Erithacus_rubecula" in present
        assert "Parus_major" in present


# ---------------------------------------------------------------------------
# Режимы отказа
# ---------------------------------------------------------------------------


def test_missing_root_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        iNatSoundsDataset(tmp_path / "does_not_exist").manifest()


def test_empty_species_dir_is_skipped(tmp_path: Path) -> None:
    root = _build_tree(tmp_path / "inat", {"Parus_major": 3})
    (root / "Empty_species").mkdir()  # папка вида без аудио

    ds = iNatSoundsDataset(root)
    # Пустой вид не попадает ни в vocab, ни в manifest.
    assert "Empty_species" not in ds.vocab()
    assert all(_species_of(r) != "Empty_species" for r in ds.manifest())


def test_single_sample_class_does_not_crash_split(tmp_path: Path) -> None:
    root = _build_tree(
        tmp_path / "inat",
        {"Parus_major": 4, "Rare_bird": 1},
    )
    ds = iNatSoundsDataset(root)

    # Класс из одного примера не должен валить стратификацию.
    train, val = ds.split(seed=3)
    assert len(train) + len(val) == 5
    all_species = {_species_of(r) for r in train} | {_species_of(r) for r in val}
    assert "Rare_bird" in all_species
