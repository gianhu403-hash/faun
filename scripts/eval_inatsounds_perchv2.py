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


def _embed_and_logits(records, cache_path: Path):
    """Эмбеддинги [N,1536] + логиты [N,Cfull]|None + метки y для набора записей.

    ОДИН forward Perch 2 на чанк даёт И эмбеддинги, И логиты (второго прохода
    нет, #F2). Предобработка — та же, что в ``Perch2Embedder``: downmix ->
    resample(32k) -> fit_window(160000) через ``faun.audio`` (единый владелец,
    ADR-0002). Эмбеддинги кэшируются в ``cache_path`` (npz через
    ``EmbeddingCache``); при совпадении длины и формы кэш переиспользуется, но
    логиты считаются заново каждый прогон (val маленький, zero-shot дешёвый).

    Возвращает ``(X[N,1536], logits[N,Cfull]|None, y[N])`` с порядком строк,
    выровненным по ``records``.

    Тяжёлый TF тянется лениво внутри ``experiments.wrappers.perch_v2.embed`` —
    вызывается через атрибут модуля.
    """
    import soundfile as sf

    from faun import audio
    import experiments.wrappers.perch_v2 as perch_v2

    y = np.asarray([rec.species for rec in records])
    ids = [rec.path for rec in records]

    # Готовим окна 32k/160000 для всех записей (та же предобработка, что у
    # Perch2Embedder; peak-normalize у эмбеддера НЕ применяется — он живёт в
    # Perch2Adapter._prepare, не в Perch2Embedder).
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

    # Попытка переиспользовать кэш эмбеддингов (только эмбеддинги; логиты всегда
    # свежие). Кэш валиден лишь когда длина и DIM совпадают с текущим набором —
    # иначе тихий рассинхрон строк/меток. Логиты считаем всё равно (нужны для
    # zero-shot на том же наборе).
    from faun.embeddings import EmbeddingCache

    cached_X = None
    if cache_path.is_file():
        try:
            cache = EmbeddingCache.load(cache_path)
            if (
                cache.embeddings.shape == (len(records), perch_v2.DIM)
                and cache.ids == ids
            ):
                cached_X = np.asarray(cache.embeddings, dtype=np.float32)
                logger.info("reusing cached embeddings: %s", cache_path)
            else:
                logger.warning(
                    "cache %s shape/ids mismatch (got %s); recomputing",
                    cache_path,
                    cache.embeddings.shape,
                )
        except ValueError as exc:
            logger.warning("ignoring corrupt cache %s: %s", cache_path, exc)

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

    # Кэшируем свежие эмбеддинги (если из кэша не пришли). Логиты не кэшируем.
    if cached_X is None:
        EmbeddingCache(embeddings=X, ids=ids, labels=list(y)).save(cache_path)
        logger.info("cached embeddings -> %s", cache_path)
    else:
        # Предпочитаем кэш как источник истины эмбеддингов (детерминизм между
        # прогонами); логиты всё равно свежие из этого forward.
        X = cached_X

    return X, all_logits, y


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
    X_train, _train_logits, y_train = _embed_and_logits(train, cache_dir / "train.npz")
    X_val, val_logits, y_val = _embed_and_logits(val, cache_dir / "val.npz")
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

    # -- 6. деплой-проба на ВСЁМ наборе (#F1) --------------------------------
    # held-out clf измеряет КАЧЕСТВО; деплой-артефакт обучается на train+val,
    # чтобы не выбрасывать 20% данных из прода.
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
    clf_full, _cv_full = train_probe_cv(X_all, y_all, seed=args.seed)
    probe_out = save_probe(clf_full, args.probe_out)
    print(f"deploy probe (trained on train+val) saved -> {probe_out}")

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
        "probe_classes": [str(c) for c in clf_full.classes_],
        "provenance": "real-eval",
        "probe_out": str(probe_out),
        "note": (
            "headline = heldout_macro_f1 on the DISJOINT val (real-eval, "
            "synthetic=False); cv_* is secondary; zero_shot_macro_f1 is the "
            "Perch 2 untrained-head baseline on the SAME val (null = cannot "
            "confirm probe superiority)."
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
