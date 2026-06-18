"""Загрузчик iNatSounds — первый источник с ИСТИННЫМИ метками видов.

iNatSounds ("iNaturalist Sounds Dataset") — аудио, разложенное по папкам видов:
``root/<species>/<audiofile>``. В отличие от raw180 (меток нет вовсе), здесь у
каждого клипа есть ground-truth вид по имени родительской папки — поэтому это
первое место, где species-level метрику можно измерить по-настоящему (но это
требует датасета на кластере; локально работаем на MINI-фикстуре).

Загрузчик — чистый stdlib + numpy, без тяжёлых зависимостей и без TensorFlow.
Эмбеддинги и обучение — отдельный слой (``faun.embeddings`` /
``train_probe_cv``), здесь только разбор дерева, словарь видов и сплит.

Точная раскладка MINI-фикстуры задокументирована в
``tests/fixtures/inatsounds_mini/README`` и совпадает с реальной.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Расширения аудио, которые считаем валидными клипами iNatSounds.
_AUDIO_SUFFIXES = frozenset({".wav", ".ogg", ".mp3", ".flac"})


@dataclass(frozen=True)
class iNatRecord:
    """Одна запись манифеста: путь к клипу + ground-truth вид."""

    path: str
    species: str


class iNatSoundsDataset:
    """Дерево ``root/<species>/<audiofile>`` -> manifest / vocab / split.

    Парсит каталог при первом обращении и кэширует результат. Виды берутся из
    имён подпапок первого уровня; пустые папки видов (без аудио) пропускаются.
    """

    def __init__(self, root) -> None:
        self._root = Path(root)
        self._manifest: list[iNatRecord] | None = None

    # -- разбор дерева ----------------------------------------------------

    def _scan(self) -> list[iNatRecord]:
        """Лениво просканировать дерево; кэшировать manifest.

        Поднимает ``NotADirectoryError`` при отсутствии/не-каталоге root —
        явный отказ вместо тихого пустого результата.
        """
        if self._manifest is not None:
            return self._manifest

        if not self._root.is_dir():
            raise NotADirectoryError(f"iNatSounds root not found: {self._root}")

        records: list[iNatRecord] = []
        # Сортируем папки и файлы — детерминированный порядок манифеста.
        for species_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
            species = species_dir.name
            for clip in sorted(species_dir.iterdir()):
                if clip.is_file() and clip.suffix.lower() in _AUDIO_SUFFIXES:
                    records.append(iNatRecord(path=str(clip), species=species))

        self._manifest = records
        return records

    # -- публичный API (FROZEN) ------------------------------------------

    def manifest(self) -> list:
        """Список ``iNatRecord(path, species)`` по всем аудиофайлам дерева."""
        return list(self._scan())

    def vocab(self) -> dict:
        """``species -> contiguous int id`` (отсортировано, детерминировано).

        Включает только виды, у которых есть хотя бы один аудиофайл (пустые
        папки видов не попадают в словарь).
        """
        species = sorted({rec.species for rec in self._scan()})
        return {name: idx for idx, name in enumerate(species)}

    def split(self, seed) -> tuple:
        """Стратифицированный ``(train, val)`` манифеста, детерминирован по seed.

        Для каждого вида клипы тасуются ГСЧ от ``seed`` и делятся ~80/20.
        Класс из одного примера целиком уходит в train (val не может быть
        пустым для остальных, а единственный пример нельзя разбить) — сплит не
        падает. Объединение train+val = весь манифест, без пересечений.
        """
        rng = np.random.default_rng(seed)
        records = self._scan()

        # Группируем по виду в детерминированном порядке имён.
        by_species: dict[str, list[iNatRecord]] = {}
        for rec in records:
            by_species.setdefault(rec.species, []).append(rec)

        train: list[iNatRecord] = []
        val: list[iNatRecord] = []
        for species in sorted(by_species):
            clips = by_species[species]
            order = rng.permutation(len(clips))
            shuffled = [clips[i] for i in order]
            if len(shuffled) < 2:
                # Единственный пример нельзя застратифицировать — в train.
                train.extend(shuffled)
                continue
            # ~20% в val, минимум 1.
            n_val = max(1, round(len(shuffled) * 0.2))
            val.extend(shuffled[:n_val])
            train.extend(shuffled[n_val:])

        return train, val
