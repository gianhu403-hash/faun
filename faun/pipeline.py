"""Reusable pipeline executor — the shared segment→classify→Detection core.

Both :func:`faun.api.run_pipeline` (single classifier; writes ``results.csv`` +
clip WAVs + ``detections.jsonl``) and :func:`faun.labeling.batch_label`
(multi-model; optional embeddings export) walk the SAME inner loop:

    for entry in manifest.entries:
        waveform, sr = read(entry.path)
        for seg in SegmentExtractor.extract(waveform, sr):
            clip  = slice(waveform, sr, seg)        # ORIGINAL sr + channels
            x16k  = downmix + resample(clip) -> 16 kHz mono
            labels = build_labels(x16k)             # caller wires the model(s)
            det = Detection.new(trap_id, source_file, segment=seg, labels=labels)

This module owns that core. :func:`run_batch` is a GENERATOR that yields one
:class:`SegmentResult` per detected segment, each carrying the ``Detection`` and
its ORIGINAL-sr clip — row-aligned by construction (``result.clip`` is the clip
for ``result.detection``). That alignment is the contract ``batch_label``'s
embeddings export depends on (ADR-0003). Yielding lazily preserves the streaming
memory profile for ``run_pipeline`` (only one clip is live at a time) while
``batch_label`` materialises the stream into a list (it needs every clip for
``embed_batch`` anyway).

Preprocessing (downmix / resample) delegates to :mod:`faun.audio` (ADR-0002).
No heavy ML lives here: classification happens through the caller-supplied
``build_labels`` callback, so the executor stays TensorFlow/torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from faun import audio
from faun.detections import Detection, Label
from faun.segmentation import Segment, SegmentExtractor

__all__ = [
    "CLASSIFY_SR",
    "SegmentResult",
    "slice_clip",
    "to_classifier_input",
    "run_batch",
]

#: Real adapters receive a 16 kHz mono array — the frozen classifier-input
#: contract (``SpeciesClassifier.classify(segment, sr)`` takes a waveform array,
#: NOT a :class:`~faun.segmentation.Segment`). See CONTEXT.md.
CLASSIFY_SR = 16_000


def slice_clip(waveform: np.ndarray, sr: int, segment: Segment) -> np.ndarray:
    """Cut ``segment``'s clip from the ORIGINAL waveform at the ORIGINAL sr.

    Channels and dtype are preserved (this clip is what gets written to disk
    and/or embedded). Indices are clamped to the waveform bounds so a segment
    that brushes the file edge never raises.
    """
    start = max(0, int(round(segment.start_s * sr)))
    end = min(len(waveform), int(round(segment.end_s * sr)))
    return waveform[start:end]


def to_classifier_input(clip: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono + resample to 16 kHz — the ``SpeciesClassifier`` input.

    The classifier protocol takes a mono float32 array at :data:`CLASSIFY_SR`,
    NOT a :class:`~faun.segmentation.Segment`; real adapters do
    ``np.asarray(segment)``, so a ``Segment`` object here would corrupt them.
    Preprocessing is delegated to :mod:`faun.audio` (the single owner).
    """
    mono = audio.downmix(clip)
    return audio.resample(mono, sr, CLASSIFY_SR)


@dataclass
class SegmentResult:
    """One detected segment: its :class:`Detection` + its aligned original clip.

    ``clip`` is the segment's waveform at ``sr`` (the ORIGINAL recording sample
    rate / channels). ``detection`` is the fully-built record. The two are
    row-aligned by construction — never reorder one without the other.
    """

    detection: Detection
    clip: np.ndarray
    sr: int


def run_batch(
    entries,
    *,
    read_waveform: Callable[[object], tuple[np.ndarray, int]],
    build_labels: Callable[[np.ndarray], list[Label]],
    extractor: SegmentExtractor | None = None,
) -> Iterator[SegmentResult]:
    """Run the shared segment→classify→Detection core over manifest entries.

    Args:
        entries: iterable of ingest ``AudioFileEntry`` (needs ``.path`` +
            ``.trap_id``), already in the desired (chronological) order.
        read_waveform: ``entry.path -> (waveform, sr)``. The caller controls the
            dtype: ``run_pipeline`` keeps float64 for clip fidelity;
            ``batch_label`` casts to float32.
        build_labels: ``16 kHz mono array -> list[Label]`` for that one segment.
            The caller wires in its classifier(s) and label provenance here.
        extractor: optional :class:`SegmentExtractor` (a fresh one by default).

    Yields:
        One :class:`SegmentResult` per detected segment, in entry/segment order,
        with ``clip`` row-aligned to ``detection`` (ADR-0003).
    """
    extractor = extractor or SegmentExtractor()
    for entry in entries:
        waveform, sr = read_waveform(entry.path)
        for seg in extractor.extract(waveform, sr):
            clip = slice_clip(waveform, sr, seg)
            labels = build_labels(to_classifier_input(clip, sr))
            det = Detection.new(
                trap_id=entry.trap_id,
                source_file=entry.path.name,
                segment=seg,
                labels=labels,
            )
            yield SegmentResult(detection=det, clip=clip, sr=sr)
