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
    STATUS_PSEUDO,
    Detection,
    Label,
    write_detections,
)
from faun.embeddings import Embedder, EmbeddingCache
from faun.ingest import scan
from faun.segmentation import SegmentExtractor

__all__ = ["batch_label", "training_candidates"]

#: Имена моделей, образующих консенсус (оба должны согласиться на вид).
_CONSENSUS_MODELS = ("perch", "birdnet")

#: source'ы, запрещённые в обучающем наборе (лицензионный гейт).
#: BirdNET = CC BY-NC-SA (non-commercial + ShareAlike).
_TRAINING_EXCLUDED_SOURCES = frozenset({SOURCE_BIRDNET})


def _model_source(name: str) -> str:
    """Имя модели -> канонический source метки (``model:<name>``)."""
    return f"model:{name}"


def _read_clip(path: Path) -> tuple[np.ndarray, int]:
    """Загрузить WAV как ``(waveform, sr)`` через soundfile (без ML)."""
    import soundfile as sf

    waveform, sr = sf.read(str(path))
    return np.asarray(waveform, dtype=np.float32), int(sr)


def _slice_segment(waveform: np.ndarray, sr: int, segment) -> np.ndarray:
    """Вырезать клип сегмента из исходного сигнала на исходной sr."""
    start = max(0, int(round(segment.start_s * sr)))
    end = min(len(waveform), int(round(segment.end_s * sr)))
    return waveform[start:end]


def _consensus_species(detection: Detection) -> set[str]:
    """Виды, предсказанные ОБЕИМИ моделями консенсуса для одной детекции."""
    by_model: dict[str, set[str]] = {}
    for name in _CONSENSUS_MODELS:
        src = _model_source(name)
        by_model[name] = {lbl.species for lbl in detection.labels if lbl.source == src}
    if any(not species for species in by_model.values()):
        return set()
    return set.intersection(*by_model.values())


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
    extractor = SegmentExtractor()

    detections: list[Detection] = []
    counts: dict[str, int] = {name: 0 for name in models}

    manifest = scan(Path(archive))
    for entry in manifest.entries:
        waveform, sr = _read_clip(entry.path)
        segments = extractor.extract(waveform, sr)
        for segment in segments:
            labels: list[Label] = []
            for name, model in models.items():
                source = _model_source(name)
                predictions: list[Prediction] = model.classify(segment, sr)
                for pred in predictions:
                    labels.append(
                        Label.from_prediction(pred, source, status=STATUS_PSEUDO)
                    )
                    counts[name] += 1
            detections.append(
                Detection.new(
                    trap_id=entry.trap_id,
                    source_file=entry.path.name,
                    segment=segment,
                    labels=labels,
                )
            )

    write_detections(out_jsonl, detections)

    n_consensus = sum(1 for det in detections if _consensus_species(det))

    summary: dict = {
        "counts": counts,
        "n_detections": len(detections),
        "n_consensus": n_consensus,
        "paths": {"detections": str(out_jsonl)},
    }

    # Экспорт эмбеддингов — только при наличии и пути, и эмбеддера.
    if emb_out is not None and embedder is not None:
        emb_out = Path(emb_out)
        rows: list[np.ndarray] = []
        ids: list[str] = []
        for det, waveform_sr in _iter_detection_clips(detections, manifest, extractor):
            waveform, sr, segment = waveform_sr
            clip = _slice_segment(waveform, sr, segment)
            rows.append(np.asarray(embedder.embed(clip, sr), dtype=np.float32))
            ids.append(det.detection_id)
        dim = int(getattr(embedder, "DIM", None) or getattr(embedder, "dim", 0))
        embeddings = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros((0, dim), dtype=np.float32)
        )
        EmbeddingCache(embeddings=embeddings, ids=ids).save(emb_out)
        summary["paths"]["embeddings"] = str(emb_out)

    return summary


def _iter_detection_clips(detections, manifest, extractor):
    """Сопоставить детекции исходным сигналам и сегментам для эмбеддинга.

    Перечитывает каждый файл один раз и переразбивает на сегменты тем же
    экстрактором, выдавая ``(detection, (waveform, sr, segment))`` в порядке
    детекций. Порядок детерминирован: детекции строились в этом же порядке.
    """
    by_file: dict[str, list[Detection]] = {}
    for det in detections:
        by_file.setdefault(det.source_file, []).append(det)

    entries_by_name = {e.path.name: e for e in manifest.entries}
    for fname, dets in by_file.items():
        entry = entries_by_name[fname]
        waveform, sr = _read_clip(entry.path)
        segments = extractor.extract(waveform, sr)
        for det, segment in zip(dets, segments):
            yield det, (waveform, sr, segment)


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
            lbl for lbl in det.labels if lbl.source not in _TRAINING_EXCLUDED_SOURCES
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
