#!/usr/bin/env python3
"""extract_inatsounds_subset.py — example regional-bird subset of the iNatSounds val split.

ЗАЧЕМ
    Прошлой ночью видовую метрику считали ad-hoc на горстке видов. Этот скрипт
    обобщает извлечение: из архива iNatSounds-val (``val.tar.gz`` + ``val.json``)
    выбирает ~40-60 видов птиц из ПРИМЕРА палеарктической лесной фауны и раскладывает их в
    дерево ``root/<Genus_species>/<clip>.wav``, которое читает
    ``faun.datasets.iNatSoundsDataset`` (та же раскладка, что в
    ``tests/fixtures/inatsounds_mini/README``).

ДВА СЛОЯ
    1. ЧИСТАЯ селекция (`select_targets`) — детерминированная, без файлов / tar /
       сети, без TensorFlow. Это и есть юнит-тестируемый контракт
       (`tests/test_inatsounds_subset.py`).
    2. ЭКСТРАКЦИЯ (`main`/CLI, под ``if __name__ == '__main__'``) — потоковое
       чтение tar, запись клипов, логирование. НЕ юнит-тестируется (нужен tar).

ГДЕ ЗАПУСКАТЬ
    На кластере внутри TF-образа (там лежит val.tar.gz). Но импорт модуля и
    `select_targets` — чистый stdlib, TF не тянется ни на каком пути.

ИНВОКАЦИЯ
    python scripts/extract_inatsounds_subset.py \
        --val-tar  /home/oleg/faun-data/datasets/inatsounds/val.tar.gz \
        --val-json /home/oleg/faun-data/datasets/inatsounds/val.json \
        --root     /home/oleg/faun-data/datasets/inatsounds_reserve \
        --n-species 50 --cap 120 --min-clips 40

ЧЕСТНОСТЬ
    Дерево на выходе несёт ИСТИННЫЕ видовые метки (имя папки = вид). Любая
    метрика по нему — реальная только при прогоне на этом реальном датасете
    (см. tests/fixtures/inatsounds_mini/README, docs/training.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
from collections import Counter

logger = logging.getLogger("extract_inatsounds_subset")

# ---------------------------------------------------------------------------
# RESERVE — ПРИМЕР-ПЛЕЙСХОЛДЕР: палеарктические лесные птицы (биномиальные имена,
# verbatim), составлен по общим знаниям. НЕ авторитетный чеклист заповедника —
# настоящий даёт орнитолог. Имя RESERVE — просто идентификатор артефакта.
# ---------------------------------------------------------------------------
RESERVE: list[str] = [
    "Fringilla coelebs",
    "Fringilla montifringilla",
    "Chloris chloris",
    "Carduelis carduelis",
    "Spinus spinus",
    "Pyrrhula pyrrhula",
    "Coccothraustes coccothraustes",
    "Emberiza citrinella",
    "Emberiza schoeniclus",
    "Turdus merula",
    "Turdus pilaris",
    "Turdus philomelos",
    "Turdus iliacus",
    "Turdus viscivorus",
    "Erithacus rubecula",
    "Luscinia luscinia",
    "Phoenicurus phoenicurus",
    "Ficedula hypoleuca",
    "Muscicapa striata",
    "Sylvia atricapilla",
    "Sylvia borin",
    "Sylvia communis",
    "Sylvia curruca",
    "Phylloscopus trochilus",
    "Phylloscopus collybita",
    "Phylloscopus sibilatrix",
    "Regulus regulus",
    "Parus major",
    "Cyanistes caeruleus",
    "Periparus ater",
    "Poecile montanus",
    "Poecile palustris",
    "Lophophanes cristatus",
    "Aegithalos caudatus",
    "Sitta europaea",
    "Certhia familiaris",
    "Troglodytes troglodytes",
    "Prunella modularis",
    "Motacilla alba",
    "Anthus trivialis",
    "Lanius collurio",
    "Sturnus vulgaris",
    "Garrulus glandarius",
    "Pica pica",
    "Corvus corax",
    "Corvus cornix",
    "Cuculus canorus",
    "Dendrocopos major",
    "Dryocopus martius",
    "Picus canus",
    "Jynx torquilla",
    "Columba palumbus",
    "Streptopelia turtur",
    "Strix aluco",
    "Asio otus",
    "Bubo bubo",
    "Glaucidium passerinum",
    "Buteo buteo",
    "Accipiter nisus",
    "Accipiter gentilis",
    "Pernis apivorus",
    "Tetrastes bonasia",
    "Grus grus",
    "Scolopax rusticola",
    "Apus apus",
    "Hirundo rustica",
    "Oriolus oriolus",
    "Anas platyrhynchos",
    "Upupa epops",
]

# Расширение валидных аудио-клипов в tar (iNatSounds-val — .wav).
_AUDIO_SUFFIX = ".wav"


# ---------------------------------------------------------------------------
# ЧИСТАЯ селекция — юнит-тестируемый контракт (без файлов / tar / сети / TF).
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    """Нормализовать имя вида: strip + схлопнуть внутренние пробелы + casefold.

    Так ``' Turdus   merula '`` и ``'turdus merula'`` совпадут с reserve.
    """
    return " ".join(name.split()).casefold()


def select_targets(
    catalog: list[dict],
    reserve: list[str],
    n_species: int,
    cap: int,
    min_clips: int,
) -> list[dict]:
    """Детерминированно выбрать <= ``n_species`` видов для извлечения.

    Args:
        catalog: записи ``{"name","audio_dir_name","clip_count"}`` (только Aves).
        reserve: биномиальные научные имена видов (пример-список, НЕ чеклист заповедника).
        n_species: верхняя граница числа выбранных видов.
        cap: переносится в каждую запись как потолок клипов/вид на ЭТАПЕ
            извлечения (на селекцию не влияет).
        min_clips: пол по числу клипов — вид с меньшим числом отбрасывается.

    Returns:
        Список записей каталога (dedup по ``name``), не длиннее ``n_species``.
        Каждая запись несёт ``clip_count >= min_clips`` и ключ ``cap``;
        дополненные top-up'ом помечены ``_topped_up=True``.

    Контракт (чистая, детерминированная функция):
        * Приоритет 1 — виды reserve: сортировка по ``clip_count`` убыв., затем
          ``name`` возр.
        * Приоритет 2 (top-up) — добор из остальных Aves (не выбранных) с
          ``clip_count >= min_clips``, по ``clip_count`` убыв., затем ``name``.
        * FAIL-LOUD (#F3): если совпадений reserve∩catalog (с порогом) меньше
          ``min(20, len(reserve)//2)`` — ERROR в лог (сколько совпало + первые 5
          имён каталога), но НЕ исключение: всё равно добираем top-up'ом.
    """
    norm_reserve = {_norm(r) for r in reserve}

    # Кандидаты по порогу клипов, dedup по нормализованному имени (детерминированно).
    qualifying: list[dict] = []
    seen: set[str] = set()
    for entry in catalog:
        if entry["clip_count"] < min_clips:
            continue
        key = _norm(entry["name"])
        if key in seen:
            continue
        seen.add(key)
        qualifying.append(entry)

    # -- Приоритет 1: reserve-совпадения --------------------------------------
    reserve_hits = [e for e in qualifying if _norm(e["name"]) in norm_reserve]
    reserve_hits.sort(key=lambda e: (-e["clip_count"], e["name"]))

    # -- FAIL-LOUD guard (#F3): мало reserve-совпадений -> ERROR, но не падаем --
    threshold = min(20, len(reserve) // 2)
    if len(reserve_hits) < threshold:
        first5 = [e["name"] for e in catalog[:5]]
        logger.error(
            "reserve∩catalog слишком мало: совпало %d (порог %d). "
            "Первые 5 имён каталога: %s. Добираю top-up'ом, не падаю.",
            len(reserve_hits),
            threshold,
            first5,
        )

    chosen: list[dict] = reserve_hits[:n_species]

    # -- Приоритет 2: top-up из остальных Aves --------------------------------
    if len(chosen) < n_species:
        chosen_names = {_norm(e["name"]) for e in chosen}
        rest = [e for e in qualifying if _norm(e["name"]) not in chosen_names]
        rest.sort(key=lambda e: (-e["clip_count"], e["name"]))
        need = n_species - len(chosen)
        for entry in rest[:need]:
            topped = dict(entry)
            topped["_topped_up"] = True
            chosen.append(topped)

    # Перенести cap в каждую запись (потолок клипов/вид на этапе извлечения).
    result: list[dict] = []
    for entry in chosen:
        out = dict(entry)
        out["cap"] = cap
        result.append(out)
    return result


# ---------------------------------------------------------------------------
# Построение каталога из val.json (чистый разбор JSON, без TF).
# ---------------------------------------------------------------------------


def build_catalog(val_json: dict) -> list[dict]:
    """Собрать каталог Aves из распарсенного ``val.json``.

    Считает аннотации на ``category_id``; оставляет только категории с
    ``supercategory == 'Aves'``. Возвращает ``[{"name","audio_dir_name",
    "clip_count"}]``, отсортированный по ``name`` (детерминированно).
    """
    counts: Counter = Counter()
    for ann in val_json.get("annotations", []):
        counts[ann["category_id"]] += 1

    catalog: list[dict] = []
    for cat in val_json.get("categories", []):
        if cat.get("supercategory") != "Aves":
            continue
        catalog.append(
            {
                "name": cat["name"],
                "audio_dir_name": cat["audio_dir_name"],
                "clip_count": counts.get(cat["id"], 0),
            }
        )
    catalog.sort(key=lambda e: e["name"])
    return catalog


# ---------------------------------------------------------------------------
# ЭКСТРАКЦИЯ — CLI/main (под __main__, НЕ юнит-тестируется: нужен tar).
# ---------------------------------------------------------------------------


def _parse_member_dir(member_name: str) -> tuple[str, str] | None:
    """Из пути члена tar ``val/<audio_dir_name>/<file>.wav`` -> (dir, basename).

    Возвращает ``None`` для всего, что не ``*/<dir>/<...>.wav`` (директории,
    не-wav, слишком короткие пути). Имя члена нормализуем через посимвольный
    split по ``/`` — не зависим от платформенного разделителя в архиве.
    """
    parts = member_name.split("/")
    if len(parts) < 2:
        return None
    basename = parts[-1]
    if not basename.lower().endswith(_AUDIO_SUFFIX):
        return None
    audio_dir_name = parts[-2]
    if not audio_dir_name:
        return None
    return audio_dir_name, basename


def extract_subset(
    val_tar: str,
    selected: list[dict],
    root: str,
    cap: int,
) -> dict[str, int]:
    """Потоково извлечь клипы выбранных видов в ``root/<Genus_species>/``.

    ПОЛНЫЙ последовательный проход по tar (архив НЕ отсортирован — НЕ делаем
    early-break). Для каждого члена: разобрать путь; если его ``audio_dir_name``
    в выбранных И вид ещё не достиг ``cap`` — записать байты в
    ``root/<name.replace(' ','_')>/<basename>``. Директории и не-wav пропускаем.

    Returns:
        ``{Genus_species: extracted_count}`` по фактически записанным клипам.
    """
    # audio_dir_name -> целевая папка Genus_species.
    dir_to_target = {e["audio_dir_name"]: e["name"].replace(" ", "_") for e in selected}
    extracted: Counter = Counter()

    # 'r|gz' — потоковый режим (не держим весь tar в памяти).
    with tarfile.open(val_tar, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = _parse_member_dir(member.name)
            if parsed is None:
                continue
            audio_dir_name, basename = parsed
            target_species = dir_to_target.get(audio_dir_name)
            if target_species is None:
                continue
            if extracted[target_species] >= cap:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:  # битый член — не падаем на одной записи.
                continue
            out_dir = os.path.join(root, target_species)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, basename)
            with open(out_path, "wb") as out_f:
                out_f.write(fobj.read())
            extracted[target_species] += 1

    return dict(extracted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Извлечь reserve-подмножество iNatSounds-val в дерево "
        "root/<Genus_species>/<clip>.wav (для iNatSoundsDataset).",
    )
    parser.add_argument("--val-tar", required=True, help="путь к val.tar.gz")
    parser.add_argument("--val-json", required=True, help="путь к val.json")
    parser.add_argument("--root", required=True, help="выходной корень дерева")
    parser.add_argument("--n-species", type=int, default=50)
    parser.add_argument("--cap", type=int, default=120)
    parser.add_argument("--min-clips", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Загрузить val.json -> каталог Aves.
    with open(args.val_json, encoding="utf-8") as f:
        val_json = json.load(f)
    catalog = build_catalog(val_json)
    logger.info("catalog: %d Aves-категорий из val.json", len(catalog))

    # 2. Чистая селекция (детерминированная).
    selected = select_targets(
        catalog,
        RESERVE,
        n_species=args.n_species,
        cap=args.cap,
        min_clips=args.min_clips,
    )

    # 3. Лог выбора: выбранные виды, выпавшие reserve, top-up.
    catalog_norm = {_norm(e["name"]) for e in catalog}
    qualifying_norm = {
        _norm(e["name"]) for e in catalog if e["clip_count"] >= args.min_clips
    }
    for e in selected:
        tag = " [top-up]" if e.get("_topped_up") else ""
        logger.info("chosen: %s (clip_count=%d)%s", e["name"], e["clip_count"], tag)

    dropped = [
        r
        for r in RESERVE
        if _norm(r) not in catalog_norm or _norm(r) not in qualifying_norm
    ]
    if dropped:
        logger.info(
            "dropped reserve (отсутствуют в каталоге или < min_clips=%d): %s",
            args.min_clips,
            dropped,
        )
    topped = [e["name"] for e in selected if e.get("_topped_up")]
    if topped:
        logger.info("topped-up (не из reserve): %s", topped)

    # 4. Извлечь (потоковый проход по tar).
    os.makedirs(args.root, exist_ok=True)
    extracted = extract_subset(args.val_tar, selected, args.root, cap=args.cap)

    # 5. Финальные per-species счётчики.
    for sp in sorted({e["name"].replace(" ", "_") for e in selected}):
        logger.info("extracted: %s -> %d clips", sp, extracted.get(sp, 0))

    total = sum(extracted.values())
    print(
        f"SUMMARY: {len(selected)} species selected, "
        f"{len(extracted)} species extracted, {total} clips total -> {args.root}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
