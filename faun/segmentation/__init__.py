"""Segmentation: onset-based segment extraction (downmix + resample) (Phase 2).

Contract (faun/INTERFACES.md, frozen):
    SegmentExtractor.extract(waveform, sr) -> list[Segment]
    Segment(start_s, duration_s)

Internally: downmix stereo -> mono (mean), resample to 16 kHz, then detect
events with the calibrated detector in ``faun.ml.onset``. The downmix/resample
primitives are delegated to :mod:`faun.audio` (single preprocessing owner,
ADR-0002); the soxr-vs-fallback choice lives there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faun import audio
from faun.ml.onset import MIN_ABSOLUTE_ENERGY, OnsetDetector

__all__ = ["Segment", "SegmentExtractor"]

#: Sample rate the onset detector is calibrated for.
TARGET_SR = 16000


@dataclass(frozen=True)
class Segment:
    """A detected event segment in the source recording (seconds)."""

    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class SegmentExtractor:
    """Extract event segments from a (possibly stereo, 48 kHz) waveform.

    Pipeline: downmix -> resample to 16 kHz -> chunked onset detection ->
    fixed-window segments with padding, clipped to file bounds.

    Args:
        energy_ratio_threshold: short/long-term energy ratio that counts
            as an onset (passed to :class:`faun.ml.onset.OnsetDetector`).
        min_segment_s: segments shorter than this are dropped.
        max_segment_s: hard cap on segment duration.
        padding_s: context included before the detected onset.
        chunk_s: chunk size fed to the stateful detector per call.

    Honest-segmentation knobs (ADR-0006), every one OFF / no-op by default so
    the produced segments — and therefore ``results.csv`` / ``detections.jsonl``
    — are byte-for-byte unchanged unless explicitly enabled:

        dense_windows: replace the transient-onset detector with a fixed grid of
            ``window_s``-long windows hopped by ``hop_s`` (recall for sustained
            song the onset detector misses). The onset path stays the default.
        window_s / hop_s: dense-grid geometry (only used when ``dense_windows``).
        silence_filter: drop produced windows whose whole-window RMS energy is at
            or below the onset detector's silence floor
            (``onset.MIN_ABSOLUTE_ENERGY``) — most useful to prune empty dense
            windows. This is whole-window RMS, an approximation of (and slightly
            more aggressive than) the detector's per-frame energy gate, not the
            same test; it is a coarse empties filter, not an onset re-derivation.
        nms_iou: when set (0, 1], greedily suppress later segments whose temporal
            IoU with an already-kept one exceeds the threshold. ``None`` keeps the
            existing construction-time "onset inside the previous segment" drop
            exactly. TEMPORAL only: ``extract`` is classifier-free (it never sees
            scores), so a per-species NMS is impossible here by construction.

    All of the above run INSIDE ``extract`` before it returns, so the clip↔
    detection row-alignment (ADR-0003) downstream is preserved — there is no
    post-``run_batch`` filtering or reordering.
    """

    def __init__(
        self,
        *,
        energy_ratio_threshold: float = 8.0,
        min_segment_s: float = 0.5,
        max_segment_s: float = 5.0,
        padding_s: float = 0.25,
        chunk_s: float = 1.0,
        dense_windows: bool = False,
        window_s: float = 5.0,
        hop_s: float = 2.5,
        silence_filter: bool = False,
        nms_iou: float | None = None,
    ) -> None:
        if min_segment_s <= 0 or max_segment_s < min_segment_s:
            raise ValueError(
                f"need 0 < min_segment_s <= max_segment_s, "
                f"got {min_segment_s=} {max_segment_s=}"
            )
        if padding_s < 0 or chunk_s <= 0:
            raise ValueError(f"invalid {padding_s=} or {chunk_s=}")
        if window_s <= 0 or hop_s <= 0:
            raise ValueError(f"need positive window_s/hop_s, got {window_s=} {hop_s=}")
        if nms_iou is not None and not 0.0 < nms_iou <= 1.0:
            raise ValueError(f"nms_iou must be in (0, 1], got {nms_iou}")
        self.energy_ratio_threshold = energy_ratio_threshold
        self.min_segment_s = min_segment_s
        self.max_segment_s = max_segment_s
        self.padding_s = padding_s
        self.chunk_s = chunk_s
        self.dense_windows = dense_windows
        self.window_s = window_s
        self.hop_s = hop_s
        self.silence_filter = silence_filter
        self.nms_iou = nms_iou

    # ------------------------------------------------------------------
    # Public API (frozen contract)
    # ------------------------------------------------------------------

    def extract(self, waveform: np.ndarray, sr: int) -> list[Segment]:
        """Return event segments, in seconds of the *original* recording.

        With every ADR-0006 knob at its default this is the original onset path,
        byte-for-byte; the dense-window / silence / NMS branches only diverge when
        explicitly enabled. Any pruning happens here before the list is returned,
        so the downstream clip↔detection alignment (ADR-0003) is never disturbed.
        """
        if sr <= 0:
            raise ValueError(f"sample rate must be positive, got {sr}")
        mono = self._downmix(np.asarray(waveform))
        if mono.size == 0:
            return []
        audio = self._resample(mono, sr)
        total_s = len(audio) / TARGET_SR
        if self.dense_windows:
            segments = self._dense_windows(total_s)
        else:
            onsets = self._detect_onsets(audio)
            segments = self._build_segments(onsets, total_s)
        if self.silence_filter:
            segments = self._drop_silent(segments, audio)
        if self.nms_iou is not None:
            segments = self._nms_temporal(segments, self.nms_iou)
        return segments

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _downmix(waveform: np.ndarray) -> np.ndarray:
        """Stereo/multichannel -> mono float32 (delegates to :mod:`faun.audio`)."""
        return audio.downmix(waveform)

    @staticmethod
    def _resample(mono: np.ndarray, sr: int) -> np.ndarray:
        """Resample mono audio to TARGET_SR (delegates to :mod:`faun.audio`)."""
        return audio.resample(mono, sr, TARGET_SR)

    def _detect_onsets(self, audio: np.ndarray) -> list[float]:
        """Chunked stateful onset detection; returns onset times in seconds."""
        detector = OnsetDetector(energy_ratio_threshold=self.energy_ratio_threshold)
        chunk_len = max(int(self.chunk_s * TARGET_SR), detector.frame_size)
        onsets: list[float] = []
        for offset in range(0, len(audio), chunk_len):
            chunk = audio[offset : offset + chunk_len]
            event = detector.detect(chunk, TARGET_SR)
            if event.triggered:
                onset_sample = offset + event.frame_index * detector.hop_size
                onsets.append(onset_sample / TARGET_SR)
        return onsets

    def _build_segments(self, onsets: list[float], total_s: float) -> list[Segment]:
        """Turn onset times into padded, clipped, non-overlapping segments."""
        segments: list[Segment] = []
        for t in sorted(onsets):
            if segments and t < segments[-1].end_s:
                continue  # already covered by the previous segment
            start = max(0.0, t - self.padding_s)
            end = min(total_s, start + self.max_segment_s)
            duration = end - start
            if duration < self.min_segment_s:
                continue
            segments.append(Segment(start_s=start, duration_s=duration))
        return segments

    def _dense_windows(self, total_s: float) -> list[Segment]:
        """Tile ``[0, total_s)`` with ``window_s`` windows hopped by ``hop_s``.

        The final window is clamped to ``total_s`` and dropped if shorter than
        ``min_segment_s``. Used for recall on sustained song the transient onset
        detector misses (FR-002). Returns windows in start order.
        """
        segments: list[Segment] = []
        if total_s <= 0:
            return segments
        start = 0.0
        while start < total_s - 1e-9:
            end = min(total_s, start + self.window_s)
            duration = end - start
            if duration >= self.min_segment_s:
                segments.append(Segment(start_s=start, duration_s=duration))
            start += self.hop_s
        return segments

    def _drop_silent(self, segments: list[Segment], audio: np.ndarray) -> list[Segment]:
        """Drop segments whose whole-window 16 kHz RMS is at/below the silence floor.

        Reuses ``faun.ml.onset.MIN_ABSOLUTE_ENERGY`` (the detector's own floor
        constant) as the threshold, so the prune is anchored to the same floor the
        onset detector uses (FR-002b). The metric here is whole-window RMS, NOT the
        detector's per-frame peak energy — a coarser, slightly more aggressive
        empties filter, used mainly to drop all-silence dense windows.
        """
        kept: list[Segment] = []
        for seg in segments:
            lo = max(0, int(round(seg.start_s * TARGET_SR)))
            hi = min(len(audio), int(round(seg.end_s * TARGET_SR)))
            frame = audio[lo:hi]
            if frame.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            if rms <= MIN_ABSOLUTE_ENERGY:
                continue
            kept.append(seg)
        return kept

    @staticmethod
    def _temporal_iou(a: Segment, b: Segment) -> float:
        """Temporal intersection-over-union of two segments (0 when disjoint)."""
        inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
        union = a.duration_s + b.duration_s - inter
        return inter / union if union > 0 else 0.0

    def _nms_temporal(self, segments: list[Segment], iou: float) -> list[Segment]:
        """Greedily suppress later segments overlapping a kept one above ``iou``.

        ``extract`` is classifier-free, so there are no scores to rank by — the
        deterministic, alignment-safe choice is to keep the earlier segment and
        drop the later overlapping ones (FR-002c). Temporal IoU only.
        """
        kept: list[Segment] = []
        for seg in segments:
            if any(self._temporal_iou(seg, k) > iou for k in kept):
                continue
            kept.append(seg)
        return kept
