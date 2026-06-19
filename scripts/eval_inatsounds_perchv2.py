#!/usr/bin/env python3
"""Честная held-out species-метрика Perch 2 + деплой-проба на iNatSounds.

НАЗНАЧЕНИЕ
    Даёт ЕДИНСТВЕННОЕ настоящее видовое число — held-out macro-F1 пробы над
    Perch 2-эмбеддингами (1536) на ДИЗЪЮНКТНОМ val-сплите iNatSounds — и
    деплой-артефакт пробы (обучен на ВСЁМ наборе). В отличие от
    ``scripts/train_inatsounds.sh`` (который оценивает на том же X, что обучал —
    утечка), здесь обучение и оценка идут на НЕПЕРЕСЕКАЮЩИХСЯ сплитах (#H2).

    Дополнительно — zero-shot baseline тем же Perch 2 на том же val (#H6): из
    SavedModel-логитов читаем 14795-классовую голову, маппим виды пробы на
    колонки логитов по научным именам (``assets/labels.csv``) и считаем
    zero-shot macro-F1 на val. Это позволяет честно сравнить «обученная проба vs
    голый zero-shot Perch 2» на одном и том же held-out наборе.

ГДЕ ЗАПУСКАТЬ
    cluster-alex, образ faun-ml-cpu/torch с ``PERCH_V2_MODEL_PATH=/models/perch2``.
    НЕ локально (нет TF, нет датасета, нет SavedModel). Тяжёлый TF тянется ЛЕНИВО
    внутри ``experiments.wrappers.perch_v2.embed`` — импорт этого модуля TF-free.

ИНВОКАЦИЯ (на кластере)
    python scripts/eval_inatsounds_perchv2.py \
        --root /home/oleg/faun-data/datasets/inatsounds \
        --probe-out /home/oleg/faun-data/models/inat_perchv2_probe.pkl \
        --cache-dir /home/oleg/faun-data/cache/perchv2_eval \
        --summary-out /home/oleg/faun-data/models/inat_perchv2_eval.json

ЧЕСТНОСТЬ
    ЕДИНСТВЕННОЕ реальное число — ``species_eval(..., synthetic=False)['macro_f1']``
    на ДИЗЪЮНКТНОМ val (заголовок отчёта). CV-CI — вторичен. Синтетику как
    реальное число НЕ печатаем (этот скрипт вообще не использует synthetic=True).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Только лёгкое здесь: numpy в requirements-pipeline.txt. Всё тяжёлое (TF, soundfile,
# sklearn, faun.embeddings/retraining/datasets) импортируется ЛЕНИВО внутри main()
# / helper'ов — импорт модуля обязан остаться TF/torch-free (CI-инвариант).
import numpy as np

logger = logging.getLogger("eval_inatsounds_perchv2")

#: Размер чанка для одного forward Perch 2 (память CPU-образа). Окна 160000
#: сэмплов float32 — ~0.6 МБ каждое, 64 окна ~40 МБ батч.
_EMBED_CHUNK = 64

#: Геометрия входа Perch 2 (32 кГц mono, окно 5 с = 160000 сэмплов). Сверяется с
#: experiments.wrappers.perch_v2.{SR,WIN_SAMPLES} в _embed_and_logits (ленивый
#: импорт), здесь — для предобработки до того, как тянуть wrapper.
_PERCH2_SR = 32_000
_PERCH2_WIN = 160_000

#: Пиковая нормировка Perch 2 — ОБЯЗАТЕЛЬНА для serve-parity. Прод обслуживает
#: пробу через PerchProbeAdapter.embed -> Perch2Adapter._prepare, который
#: нормирует окно к пику 0.25 ПЕРЕД инференсом. Если обучить пробу на
#: НЕнормированных эмбеддингах (как делает Perch2Embedder), а в проде кормить
#: нормированными — train/serve skew тихо просадит точность задеплоенной пробы.
#: Поэтому здесь применяем ту же нормировку, что Perch2Adapter._prepare, и к
#: эмбеддингам (обучение пробы), и к логитам (zero-shot) — оба пути совпадут с
#: тем, что реально считает прод.
_PERCH2_PEAK = 0.25

#: Известные header-токены в Perch 2 label-ассете (первая строка именует
#: таксономию/namespace, НЕ класс). Зеркалит faun.classification.perch_v2.
_LABELS_HEADER_SENTINELS = frozenset({"inat2024_fsd50k", "ebird2021"})
_PERCH_V2_LABELS_FILE = "labels.csv"

#: Тег предобработки эмбеддингов — часть идентичности кэша (#SF-1). Если кэш
#: записан с иной нормировкой, переиспользовать его НЕЛЬЗЯ (тихий train/serve
#: skew). Пишется в sidecar ``<cache>.preproc``; на загрузке сверяется. Бэк-компат:
#: отсутствие sidecar (старый кэш) → warning + reuse (этот скрипт всегда
#: peak-normalize, так что старый кэш этого же скрипта валиден).
_PREPROC_TAG = "downmix-resample32k-fitwin160000-peak0.25"


def _binomial(species: str) -> str:
    """``Genus_species`` (имя папки iNatSounds) -> ``Genus species`` (биномиал).

    Дерево датасета хранит вид как ``Genus_species`` (подчёркивание), а и
    ``assets/labels.csv`` Perch 2, и список RESERVE, и прод-вывод используют
    биномиал с пробелом. Без этой нормализации (а) проба обучилась бы на классах
    ``Fringilla_coelebs`` — заказчик увидел бы подчёркивания, и (б) zero-shot
    маппинг по именам в labels.csv дал бы НОЛЬ совпадений (тихо null baseline).
    """
    return species.replace("_", " ")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Честная held-out species-метрика Perch 2 + деплой-проба на "
            "iNatSounds (обучение и оценка на дизъюнктных сплитах)."
        ),
    )
    p.add_argument(
        "--root",
        required=True,
        help="Корень iNatSounds: дерево root/<species>/<clip>.",
    )
    p.add_argument(
        "--probe-out",
        required=True,
        help="Куда сохранить деплой-пробу (pickle, обучена на ВСЁМ наборе).",
    )
    p.add_argument(
        "--cache-dir",
        required=True,
        help="Каталог для npz-кэшей эмбеддингов (отдельный файл на сплит).",
    )
    p.add_argument(
        "--summary-out",
        required=True,
        help="Куда записать JSON-сводку метрик.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для сплита и кросс-валидации (по умолчанию 42).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Сколько per-species чисел печатать в человеческий stdout (5).",
    )
    return p.parse_args(argv)


def _embed_and_logits(records, cache_path: Path, *, needs_logits: bool):
    """Эмбеддинги [N,1536] + логиты [N,Cfull]|None + метки y для набора записей.

    ОДИН forward Perch 2 на чанк даёт И эмбеддинги, И логиты (второго прохода
    нет, #F2). Предобработка — та же, что в ``Perch2Embedder`` ПЛЮС peak-normalize
    к 0.25 (serve-parity с ``Perch2Adapter._prepare``, см. ниже): downmix ->
    resample(32k) -> fit_window(160000) -> peak-norm через ``faun.audio``.

    Кэш (#CR-1, #SF-1): эмбеддинги кэшируются в ``cache_path`` (npz через
    ``EmbeddingCache``) + sidecar ``<cache>.preproc`` с тегом предобработки.
    На загрузке: shape+ids+preproc-тег должны совпасть. Если ``needs_logits``
    False (train-путь) и кэш валиден — forward НЕ запускается вовсе (train.npz
    реально экономит ~40 мин). Если ``needs_logits`` True (val-путь, нужны логиты
    для zero-shot) — forward всё равно идёт (логиты не кэшируются), но эмбеддинги
    берутся из кэша как источник истины.

    ``y`` нормализуется в биномиал (``Genus_species`` -> ``Genus species``), чтобы
    классы пробы, RESERVE и labels.csv жили в одном пространстве имён.

    Возвращает ``(X[N,1536], logits[N,Cfull]|None, y[N])``, строки выровнены по
    ``records``. Тяжёлый TF тянется лениво внутри wrapper'а.
    """
    import soundfile as sf

    from faun import audio
    import experiments.wrappers.perch_v2 as perch_v2

    y = np.asarray([_binomial(rec.species) for rec in records])
    ids = [rec.path for rec in records]
    preproc_sidecar = cache_path.with_suffix(".preproc")

    # -- попытка переиспользовать кэш эмбеддингов (#CR-1/#SF-1) ----------------
    cached_X = _load_cached_embeddings(cache_path, preproc_sidecar, ids, perch_v2.DIM)
    if cached_X is not None and not needs_logits:
        # train-путь: эмбеддинги есть, логиты не нужны → forward пропускаем.
        logger.info("cache hit + logits not needed → skipping Perch 2 forward")
        return cached_X, None, y

    # Готовим окна 32k/160000 (та же предобработка, что обслуживает прод).
    windows = []
    for rec in records:
        wav, sr = sf.read(rec.path)
        mono = audio.downmix(np.asarray(wav, dtype=np.float32))
        resampled = audio.resample(mono, int(sr), perch_v2.SR)
        window = audio.fit_window(resampled, perch_v2.WIN_SAMPLES)
        # Serve-parity (#skew): Perch2Adapter._prepare нормирует окно к пику 0.25
        # перед инференсом, и именно так прод обслуживает PerchProbeAdapter. Без
        # этого проба обучилась бы на НЕнормированных эмбеддингах, а в проде ела
        # бы нормированные — тихая просадка. Нормируем И эмбеддинги, И логиты.
        peak = float(np.max(np.abs(window))) if window.size else 0.0
        if peak > 0.0:
            window = (window / peak) * _PERCH2_PEAK
        windows.append(np.asarray(window, dtype=np.float32))

    if not windows:
        # Пустой набор: согласованные пустые формы (Cfull неизвестен → 0 колонок).
        return (
            np.zeros((0, perch_v2.DIM), dtype=np.float32),
            None,
            y,
        )

    batch = np.stack(windows).astype(np.float32)  # [N, 160000]

    # Один forward на чанк: и эмбеддинги, и логиты (избегаем второго прохода, #F2).
    emb_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    logits_available = True
    for i in range(0, len(batch), _EMBED_CHUNK):
        emb, logits = perch_v2.embed(batch[i : i + _EMBED_CHUNK])
        emb_parts.append(np.asarray(emb, dtype=np.float32))
        if logits is None:
            logits_available = False
        elif logits_available:
            logit_parts.append(np.asarray(logits))

    X = np.concatenate(emb_parts, axis=0).astype(np.float32)
    all_logits = (
        np.concatenate(logit_parts, axis=0)
        if (logits_available and logit_parts)
        else None
    )

    # Источник истины эмбеддингов — кэш (если валиден); логиты всегда свежие.
    if cached_X is not None:
        X = cached_X
    else:
        from faun.embeddings import EmbeddingCache

        EmbeddingCache(embeddings=X, ids=ids, labels=list(y)).save(cache_path)
        preproc_sidecar.write_text(_PREPROC_TAG, encoding="utf-8")
        logger.info("cached embeddings -> %s (preproc=%s)", cache_path, _PREPROC_TAG)

    return X, all_logits, y


def _load_cached_embeddings(cache_path, preproc_sidecar, ids, dim):
    """Вернуть кэш-эмбеддинги [N,dim] если валиден, иначе ``None`` (#CR-1/#SF-1).

    Валидность: файл читается, ``shape == (len(ids), dim)``, ``ids`` совпадают,
    и preproc-тег sidecar совпадает с :data:`_PREPROC_TAG`. Отсутствие sidecar
    (старый кэш этого же скрипта) — warning + reuse (бэк-компат: скрипт всегда
    peak-normalize). Несовпадение тега — recompute (защита от train/serve skew).
    """
    from faun.embeddings import EmbeddingCache

    if not Path(cache_path).is_file():
        return None
    try:
        cache = EmbeddingCache.load(cache_path)
    except ValueError as exc:
        logger.warning("ignoring corrupt cache %s: %s", cache_path, exc)
        return None
    if cache.embeddings.shape != (len(ids), dim) or cache.ids != ids:
        logger.warning(
            "cache %s shape/ids mismatch (got %s); recomputing",
            cache_path,
            cache.embeddings.shape,
        )
        return None
    sidecar = Path(preproc_sidecar)
    if sidecar.is_file():
        tag = sidecar.read_text(encoding="utf-8").strip()
        if tag != _PREPROC_TAG:
            logger.warning(
                "cache %s preproc tag %r != %r; recomputing (avoid train/serve skew)",
                cache_path,
                tag,
                _PREPROC_TAG,
            )
            return None
    else:
        logger.warning(
            "cache %s has no preproc sidecar; reusing (back-compat — this script "
            "always peak-normalizes)",
            cache_path,
        )
    logger.info("reusing cached embeddings: %s", cache_path)
    return np.asarray(cache.embeddings, dtype=np.float32)


def _load_perch2_labels(model_path: str | None):
    """Прочитать научные имена классов из ``<model_path>/assets/labels.csv``.

    Зеркалит header-drop логику ``faun.classification.perch_v2._load_labels``:
    дропаем ведущую строку, если она в ``_LABELS_HEADER_SENTINELS`` ИЛИ без
    внутреннего пробела (реальные имена биномиальны «Genus species»). Возвращает
    ``list[str]`` (имя -> позиция = колонка логита) либо ``None`` при отсутствии
    ассета / пустом файле. Чистый file-I/O, TF не импортирует.
    """
    if not model_path:
        return None
    assets = Path(model_path) / "assets" / _PERCH_V2_LABELS_FILE
    if not assets.is_file():
        logger.warning(
            "Perch 2 labels file not found at %s; zero-shot baseline disabled",
            assets,
        )
        return None
    try:
        text = assets.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("failed to read Perch 2 labels %s: %s", assets, exc)
        return None
    rows = [ln.split(",")[0].strip() for ln in text.splitlines() if ln.strip()]
    if not rows:
        logger.warning("Perch 2 labels file %s is empty; zero-shot disabled", assets)
        return None
    first = rows[0]
    if first in _LABELS_HEADER_SENTINELS or " " not in first:
        rows = rows[1:]
    return rows or None


def _zero_shot_macro_f1(y_val, val_logits, model_path, probe_classes):
    """Zero-shot Perch 2 macro-F1 на val + покрытие (#H6).

    Маппит каждый класс пробы (``probe_classes``) на колонку логита по научному
    имени (``assets/labels.csv``), ограничивает val-логиты этими колонками,
    берёт argmax -> имя, и считает ``f1_score(average='macro')`` на val.

    Возвращает ``(zero_shot_macro_f1: float|None, coverage: int)``. ``None``
    (gate читает как «не могу подтвердить превосходство пробы») когда: логитов
    нет; ассет меток отсутствует/пуст; на колонки маппится < 2 классов пробы.
    """
    if val_logits is None:
        logger.warning("no Perch 2 logits on val; zero-shot baseline = null")
        return None, 0

    label_names = _load_perch2_labels(model_path)
    if not label_names:
        return None, 0

    name_to_col = {name: col for col, name in enumerate(label_names)}
    # Класс пробы -> колонка логита (только пересечение словарей).
    mapped = {
        cls: name_to_col[str(cls)] for cls in probe_classes if str(cls) in name_to_col
    }
    coverage = len(mapped)
    if coverage < 2:
        logger.warning(
            "only %d/%d probe classes map to Perch 2 logit columns; "
            "zero-shot baseline = null",
            coverage,
            len(probe_classes),
        )
        return None, coverage

    val_logits = np.asarray(val_logits)
    mapped_classes = list(mapped.keys())
    cols = [mapped[cls] for cls in mapped_classes]
    restricted = val_logits[:, cols]  # [N_val, coverage]
    pred_idx = np.argmax(restricted, axis=1)
    preds = np.asarray([mapped_classes[i] for i in pred_idx])

    from sklearn.metrics import f1_score

    score = float(f1_score(np.asarray(y_val), preds, average="macro", zero_division=0))
    return score, coverage


def _gate_comparison(clf, X_val, y_val, val_logits, model_path):
    """Apples-to-apples проба-vs-zero-shot на ОДНОМ пространстве меток (#COMP-2/ARCH-2).

    Полный ``_zero_shot_macro_f1`` штрафует baseline за виды, которых нет в
    eBird-таксономии Perch 2 (структурно недостижимы) — это честно как «цена
    покрытия», но НЕ годится как число гейта: проба сравнивалась бы с искусственно
    ослабленным baseline. Здесь ОБА (проба и zero-shot) считаются на ПЕРЕСЕЧЕНИИ:
    виды, которые (а) есть в val И (б) маппятся на колонку логита Perch 2. На этом
    наборе и проба, и zero-shot выбирают из одних и тех же кандидатов — сравнение
    корректно.

    Возвращает dict ``{gate_n, gate_species_n, gate_probe_macro_f1,
    gate_zero_shot_macro_f1}`` либо ``None`` (логитов нет / меток нет / пересечение
    < 2 видов — гейт читает None как «не могу подтвердить превосходство»).
    """
    if val_logits is None:
        return None
    label_names = _load_perch2_labels(model_path)
    if not label_names:
        return None

    name_to_col = {name: col for col, name in enumerate(label_names)}
    val_species = set(map(str, np.asarray(y_val).tolist()))
    # Пересечение: класс пробы, который И в val, И маппится на колонку логита.
    inter = [
        str(c) for c in clf.classes_ if str(c) in name_to_col and str(c) in val_species
    ]
    if len(inter) < 2:
        logger.warning(
            "gate intersection < 2 species (%d); gate comparison = null", len(inter)
        )
        return None

    inter_set = set(inter)
    mask = np.array([str(s) in inter_set for s in np.asarray(y_val)])
    if not mask.any():
        return None

    y_inter = np.asarray(y_val)[mask]
    classes = list(map(str, clf.classes_))
    probe_cols = [classes.index(c) for c in inter]
    proba = np.asarray(clf.predict_proba(np.asarray(X_val)[mask]))[:, probe_cols]
    probe_pred = np.asarray([inter[i] for i in np.argmax(proba, axis=1)])

    logit_cols = [name_to_col[c] for c in inter]
    zs = np.asarray(val_logits)[mask][:, logit_cols]
    zs_pred = np.asarray([inter[i] for i in np.argmax(zs, axis=1)])

    from sklearn.metrics import f1_score

    return {
        "gate_n": int(mask.sum()),
        "gate_species_n": len(inter),
        "gate_probe_macro_f1": float(
            f1_score(y_inter, probe_pred, average="macro", zero_division=0)
        ),
        "gate_zero_shot_macro_f1": float(
            f1_score(y_inter, zs_pred, average="macro", zero_division=0)
        ),
    }


def _coerce(value):
    """numpy-скаляры -> питоновские float/int для json.dump."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)

    # Ленивые тяжёлые импорты (модуль обязан остаться TF-free на import).
    from faun.datasets import iNatSoundsDataset
    from faun.retraining import save_probe, species_eval, train_probe_cv
    from sklearn.metrics import accuracy_score

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. сплит + инвариант дизъюнктности (#H2) ----------------------------
    ds = iNatSoundsDataset(args.root)
    train, val = ds.split(args.seed)
    train_paths = {rec.path for rec in train}
    val_paths = {rec.path for rec in val}
    if not train_paths.isdisjoint(val_paths):
        raise SystemExit(
            "FATAL: train/val splits overlap — held-out metric would leak. "
            "Aborting (this is the #H2 honesty invariant)."
        )
    if len(val) == 0:
        raise SystemExit(
            "FATAL: empty val split — cannot measure a held-out species metric. "
            "Need >= 2 clips per species for a stratified hold-out."
        )

    n_species = len({rec.species for rec in train} | {rec.species for rec in val})
    print(
        f"split(seed={args.seed}): {len(train)} train / {len(val)} val clips, "
        f"{n_species} species (disjoint OK)"
    )

    # -- 2. эмбеддинги ПО СПЛИТАМ, отдельный forward на сплит (#F2 без утечки) --
    # train и val эмбеддятся РАЗДЕЛЬНО, каждый в свой npz-кэш под cache-dir.
    # train: логиты не нужны (обучаем пробу на эмбеддингах) → кэш экономит forward
    # (#CR-1). val: логиты нужны для zero-shot → forward всё равно идёт.
    X_train, _train_logits, y_train = _embed_and_logits(
        train, cache_dir / "train.npz", needs_logits=False
    )
    X_val, val_logits, y_val = _embed_and_logits(
        val, cache_dir / "val.npz", needs_logits=True
    )
    print(
        f"embeddings: train X={X_train.shape}, val X={X_val.shape} "
        f"(val logits: {'present' if val_logits is not None else 'none'})"
    )

    # -- 3. обучение ТОЛЬКО на train (#H2) -----------------------------------
    clf, cv = train_probe_cv(X_train, y_train, seed=args.seed)
    print(
        f"train_probe_cv (train only): metric={cv['metric']} "
        f"value={cv['value']:.4f} ci=[{cv['ci_low']},{cv['ci_high']}] "
        f"n={cv['n']} n_classes={cv['n_classes']}"
    )

    # -- 4. held-out species-метрика на ДИЗЪЮНКТНОМ val (#H2) -----------------
    # ЕДИНСТВЕННОЕ реальное число — synthetic=False. Заголовок — report['macro_f1']
    # на val (НЕ report['value'], которое = CV на val-наборе).
    report = species_eval(clf, X_val, y_val, synthetic=False)
    heldout_macro_f1 = float(report["macro_f1"])
    heldout_accuracy = float(accuracy_score(np.asarray(y_val), clf.predict(X_val)))
    print(
        f"species_eval(synthetic=False) HELD-OUT: macro_f1={heldout_macro_f1:.4f} "
        f"accuracy={heldout_accuracy:.4f} provenance={report['provenance']}"
    )
    print(f"per_species_recall (top-{args.top_k} lowest):")
    worst = sorted(report["per_species_recall"].items(), key=lambda kv: kv[1])
    for sp, rec in worst[: args.top_k]:
        print(f"  {sp}: {rec:.3f}")

    # -- 5. zero-shot baseline на ТОМ ЖЕ val (#H2, #H6) ----------------------
    import os

    model_path = os.environ.get("PERCH_V2_MODEL_PATH")
    zero_shot_macro_f1, zero_shot_coverage = _zero_shot_macro_f1(
        y_val, val_logits, model_path, clf.classes_
    )
    if zero_shot_macro_f1 is None:
        print(
            "zero_shot baseline: null "
            f"(coverage={zero_shot_coverage} probe classes mapped to logits) — "
            "cannot confirm probe superiority"
        )
    else:
        print(
            f"zero_shot baseline (Perch 2 head, same val): "
            f"macro_f1={zero_shot_macro_f1:.4f} "
            f"coverage={zero_shot_coverage}/{len(clf.classes_)} classes"
        )

    # -- 5b. apples-to-apples гейт-сравнение на пересечении (#COMP-2/ARCH-2) --
    # Для РЕШЕНИЯ о выкатке используем числа на ОДНОМ пространстве меток (виды в
    # val, маппящиеся на колонку логита Perch 2), а не полный-val baseline,
    # который структурно занижен.
    gate = _gate_comparison(clf, X_val, y_val, val_logits, model_path)
    if gate is None:
        print("gate comparison: null (cannot fairly compare probe vs zero-shot)")
    else:
        print(
            f"gate (intersection, {gate['gate_species_n']} species, n={gate['gate_n']}): "
            f"probe_macro_f1={gate['gate_probe_macro_f1']:.4f} vs "
            f"zero_shot_macro_f1={gate['gate_zero_shot_macro_f1']:.4f}"
        )

    # -- 6. деплой-проба на ВСЁМ наборе (#F1) --------------------------------
    # held-out clf измеряет КАЧЕСТВО; деплой-артефакт обучается на train+val,
    # чтобы не выбрасывать 20% данных из прода.
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
    clf_full, _cv_full = train_probe_cv(X_all, y_all, seed=args.seed)
    probe_out = save_probe(clf_full, args.probe_out)
    print(f"deploy probe (trained on train+val) saved -> {probe_out}")

    # -- 6b. measured == deployed гейт (#COMP-1) -----------------------------
    # Заголовочное число измерено на видах val; деплой-проба обучена на train+val.
    # Если у пробы есть классы, которых НЕ было в val (одно-клиповые виды целиком
    # ушли в train), held-out число их НЕ покрывает — заказчику об этом честно.
    measured_species = sorted(report["per_species_recall"].keys())
    deployed_species = sorted(str(c) for c in clf_full.classes_)
    unmeasured = sorted(set(deployed_species) - set(measured_species))
    if unmeasured:
        logger.warning(
            "deploy probe serves %d species NOT measured on val (single-clip → "
            "all-train): %s — held-out macro-F1 does not cover them",
            len(unmeasured),
            unmeasured,
        )

    # -- 7. JSON-сводка ------------------------------------------------------
    summary = {
        "n_species": int(n_species),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "cv_metric": cv["metric"],
        "cv_value": _coerce(cv["value"]),
        "cv_ci_low": _coerce(cv["ci_low"]),
        "cv_ci_high": _coerce(cv["ci_high"]),
        "heldout_macro_f1": heldout_macro_f1,
        "heldout_accuracy": heldout_accuracy,
        "per_species_recall": {
            str(k): _coerce(v) for k, v in report["per_species_recall"].items()
        },
        "zero_shot_macro_f1": (
            None if zero_shot_macro_f1 is None else float(zero_shot_macro_f1)
        ),
        "zero_shot_coverage": int(zero_shot_coverage),
        # apples-to-apples гейт-числа на пересечении (#COMP-2/ARCH-2): именно их
        # сравнивает V3-гейт (probe >= zero-shot), не полный-val baseline.
        "gate_probe_macro_f1": (None if gate is None else gate["gate_probe_macro_f1"]),
        "gate_zero_shot_macro_f1": (
            None if gate is None else gate["gate_zero_shot_macro_f1"]
        ),
        "gate_species_n": (None if gate is None else gate["gate_species_n"]),
        "gate_n": (None if gate is None else gate["gate_n"]),
        "probe_classes": deployed_species,
        "measured_species": measured_species,
        "unmeasured_deployed_species": unmeasured,
        "provenance": "real-eval",
        "probe_out": str(probe_out),
        "note": (
            "headline = heldout_macro_f1 on the DISJOINT val (real-eval, "
            "synthetic=False); cv_* is the TRAIN-only CV (secondary, defensible). "
            "V3 gate compares gate_probe_macro_f1 vs gate_zero_shot_macro_f1 "
            "(apples-to-apples on the val species that map to a Perch 2 logit "
            "column); plain zero_shot_macro_f1 is over full val (coverage-honest "
            "but pessimistic). CAVEAT (DEVIL-1/COMP-3): embeddings are the first "
            "5 s of each iNat focal clip (fit_window left-crop), whereas prod "
            "feeds an onset-DETECTED segment — a whole-clip-vs-onset domain shift, "
            "so this number characterises probe quality on iNat heads, not raw180 "
            "serving accuracy. species labels are binomial (Genus species) to "
            "match RESERVE/labels.csv and the served name space."
        ),
    }
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"summary JSON -> {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
