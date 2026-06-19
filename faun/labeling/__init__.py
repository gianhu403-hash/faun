"""Батч-разметка: архив ловушек -> детекции с псевдо-метками моделей.

Контур :func:`batch_label`:

    scan(archive) -> per-file сегменты (SegmentExtractor) -> на каждый сегмент
    каждая модель из ``models`` даёт предсказания -> детекция с псевдо-метками
    (source=model:<name>, status=pseudo). Опционально считается консенсус
    Perch∩BirdNET и экспортируются эмбеддинги (один на детекцию).

Результат — ``detections.jsonl`` (через :func:`faun.detections.write_detections`)
плюс сводка-словарь.

ЛИЦЕНЗИОННЫЙ ГЕЙТ (merge-blocker). BirdNET — CC BY-NC-SA: non-commercial +
ShareAlike. Псевдо-метки BirdNET годятся ТОЛЬКО для инвентаризации/приоритизации
для эксперта и НИКОГДА не идут в обучающий набор коммерческой модели —
ShareAlike "заражает" дообученную голову. :func:`training_candidates` жёстко
выкидывает любые метки ``model:birdnet``. См. ``docs/labeling.md``.

stdlib + numpy; тяжёлый ML — только через переданные модели/эмбеддер (ленивый TF
внутри их адаптеров). Сам модуль TF не импортирует.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from faun.classification import Prediction, SpeciesClassifier
from faun.detections import (
    SOURCE_BIRDNET,
    SOURCE_PERCH,
    SOURCE_PERCH_V2,
    STATUS_PSEUDO,
    TRAINING_EXCLUDED_SOURCES,
    Detection,
    Label,
    write_detections,
)
from faun.embeddings import Embedder, EmbeddingCache, _embedder_dim, embed_batch
from faun.ingest import scan
from faun.pipeline import run_batch

__all__ = ["batch_label", "training_candidates"]

#: Классификаторы получают mono @ 16 кГц — тот же контракт SpeciesClassifier,
#: что и в ``faun.api.run_pipeline`` (НЕ сам объект Segment).
CLASSIFY_SR = 16_000

#: Консенсус-арки: детекция «в консенсусе», когда КАЖДАЯ арка дала вид и виды
#: пересекаются. Арка Perch принимает Perch 1 ИЛИ Perch 2 — оператор может
#: запустить любой (`--models perch-v2,birdnet`), и консенсус не должен молча
#: обнуляться из-за смены имени модели. Каждая арка — множество допустимых source.
_CONSENSUS_ARMS: tuple[frozenset[str], ...] = (
    frozenset({SOURCE_PERCH, SOURCE_PERCH_V2}),  # арка Perch (v1 или v2)
    frozenset({SOURCE_BIRDNET}),  # арка BirdNET
)


def _model_source(name: str) -> str:
    """Имя модели -> канонический source метки (``model:<name>``)."""
    return f"model:{name}"


def _read_clip(path: Path) -> tuple[np.ndarray, int]:
    """Загрузить WAV как ``(waveform, sr)`` через soundfile (без ML)."""
    import soundfile as sf

    waveform, sr = sf.read(str(path))
    return np.asarray(waveform, dtype=np.float32), int(sr)


def _consensus_species(detection: Detection) -> set[str]:
    """Виды, на которых сошлись ВСЕ консенсус-арки (Perch∩BirdNET) для детекции.

    Арка Perch матчит и ``model:perch`` (Perch 1), и ``model:perch-v2`` (Perch 2),
    так что выбор модели оператором не обнуляет консенсус молча.
    """
    by_arm: list[set[str]] = []
    for arm_sources in _CONSENSUS_ARMS:
        by_arm.append(
            {lbl.species for lbl in detection.labels if lbl.source in arm_sources}
        )
    if any(not species for species in by_arm):
        return set()
    return set.intersection(*by_arm)


def batch_label(
    archive,
    models: Mapping[str, SpeciesClassifier],
    out_jsonl,
    emb_out=None,
    embedder: Embedder | None = None,
) -> dict:
    """Прогнать батч-разметку архива и записать ``detections.jsonl``.

    Args:
        archive: корень архива ловушек (папка A1..A5) — сканируется
            :func:`faun.ingest.scan`.
        models: отображение ``{"perch": <clf>, "birdnet": <clf>}``; каждый
            адаптер реализует протокол :class:`SpeciesClassifier`.
        out_jsonl: путь для ``detections.jsonl`` (пишется всегда, в т.ч. пустой).
        emb_out: путь для ``embeddings.npz`` (один эмбеддинг на детекцию);
            записывается только если задан И передан ``embedder``.
        embedder: :class:`faun.embeddings.Embedder` для экспорта эмбеддингов.

    Returns:
        Сводка::

            {
              "counts": {"perch": <n_меток>, ...},
              "n_detections": <int>,
              "n_consensus": <int детекций с Perch∩BirdNET>,
              "paths": {"detections": "...", ["embeddings": "..."]},
            }
    """
    out_jsonl = Path(out_jsonl)

    counts: dict[str, int] = {name: 0 for name in models}

    def _build_labels(classifier_input: np.ndarray) -> list[Label]:
        """Map a 16 kHz mono clip through every model -> pseudo-labels."""
        labels: list[Label] = []
        for name, model in models.items():
            source = _model_source(name)
            predictions: list[Prediction] = model.classify(
                classifier_input, CLASSIFY_SR
            )
            for pred in predictions:
                labels.append(Label.from_prediction(pred, source, status=STATUS_PSEUDO))
                counts[name] += 1
        return labels

    from faun.sources import resolve_source

    # archive may be a local dir, an http(s) zip URL, or a Yandex.Disk share.
    scan_dir = resolve_source(str(archive), out_jsonl.parent)
    manifest = scan(scan_dir)

    # Single shared executor pass (faun.pipeline.run_batch): detections + their
    # original-sr clips (waveform, sr) row-aligned for the optional embeddings
    # export. Reading at float32 here keeps batch_label's prior dtype behaviour.
    results = list(
        run_batch(
            manifest.entries, read_waveform=_read_clip, build_labels=_build_labels
        )
    )
    detections: list[Detection] = [r.detection for r in results]
    clips: list[tuple[np.ndarray, int]] = [(r.clip, r.sr) for r in results]

    write_detections(out_jsonl, detections)

    n_consensus = sum(1 for det in detections if _consensus_species(det))

    summary: dict = {
        "counts": counts,
        "n_detections": len(detections),
        "n_consensus": n_consensus,
        "paths": {"detections": str(out_jsonl)},
    }

    # Экспорт эмбеддингов — только при наличии и пути, и эмбеддера. Переиспользуем
    # единый владелец ``embed_batch`` (без повторного чтения/сегментации файлов;
    # строки выровнены с detections по порядку сбора).
    if emb_out is not None and embedder is not None:
        emb_out = Path(emb_out)
        embeddings = (
            embed_batch(clips, embedder)
            if clips
            else np.zeros((0, _embedder_dim(embedder)), dtype=np.float32)
        )
        ids = [det.detection_id for det in detections]
        EmbeddingCache(embeddings=embeddings, ids=ids).save(emb_out)
        summary["paths"]["embeddings"] = str(emb_out)

    return summary


def training_candidates(detections) -> list:
    """Детекции с метками, допустимыми в обучающий набор (лицензионный гейт).

    Возвращает копии детекций, из которых вычищены любые метки с запрещённым
    source — в первую очередь ``model:birdnet`` (CC BY-NC-SA: non-commercial +
    ShareAlike). Детекции без единой допустимой метки в результат не попадают.

    Это hard-гейт коммерческого продукта: псевдо-метки BirdNET годятся только
    для приоритизации эксперту, но НИКОГДА для дообучения продовой головы.
    """
    result: list[Detection] = []
    for det in detections:
        kept = [
            lbl for lbl in det.labels if lbl.source not in TRAINING_EXCLUDED_SOURCES
        ]
        if not kept:
            continue
        result.append(
            Detection(
                detection_id=det.detection_id,
                trap_id=det.trap_id,
                source_file=det.source_file,
                segment=det.segment,
                segment_path=det.segment_path,
                labels=kept,
            )
        )
    return result
