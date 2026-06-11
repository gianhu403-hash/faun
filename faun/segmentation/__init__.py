"""Segmentation: onset-based segment extraction (downmix + resample) (Phase 2).

Contract (faun/INTERFACES.md, frozen):
    SegmentExtractor.extract(waveform, sr) -> list[Segment]
    Segment(start_s, duration_s)

Internally: downmix stereo -> mono (mean), resample to 16 kHz (soxr, with a
plain decimation/interpolation fallback), then detect events with the
calibrated detector in ``faun.ml.onset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faun.ml.onset import OnsetDetector

try:
    import soxr

    _HAS_SOXR = True
except ImportError:  # pragma: no cover - soxr is in requirements-pipeline.txt
    _HAS_SOXR = False

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
    """

    def __init__(
        self,
        *,
        energy_ratio_threshold: float = 8.0,
        min_segment_s: float = 0.5,
        max_segment_s: float = 5.0,
        padding_s: float = 0.25,
        chunk_s: float = 1.0,
    ) -> None:
        if min_segment_s <= 0 or max_segment_s < min_segment_s:
            raise ValueError(
                f"need 0 < min_segment_s <= max_segment_s, "
                f"got {min_segment_s=} {max_segment_s=}"
            )
        if padding_s < 0 or chunk_s <= 0:
            raise ValueError(f"invalid {padding_s=} or {chunk_s=}")
        self.energy_ratio_threshold = energy_ratio_threshold
        self.min_segment_s = min_segment_s
        self.max_segment_s = max_segment_s
        self.padding_s = padding_s
        self.chunk_s = chunk_s

    # ------------------------------------------------------------------
    # Public API (frozen contract)
    # ------------------------------------------------------------------

    def extract(self, waveform: np.ndarray, sr: int) -> list[Segment]:
        """Return event segments, in seconds of the *original* recording."""
        if sr <= 0:
            raise ValueError(f"sample rate must be positive, got {sr}")
        mono = self._downmix(np.asarray(waveform))
        if mono.size == 0:
            return []
        audio = self._resample(mono, sr)
        total_s = len(audio) / TARGET_SR
        onsets = self._detect_onsets(audio)
        return self._build_segments(onsets, total_s)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _downmix(waveform: np.ndarray) -> np.ndarray:
        """Stereo/multichannel -> mono float32 via channel mean."""
        if waveform.ndim == 1:
            return waveform.astype(np.float32, copy=False)
        if waveform.ndim != 2:
            raise ValueError(f"expected 1-D or 2-D waveform, got ndim={waveform.ndim}")
        # soundfile yields (frames, channels); also accept (channels, frames).
        channel_axis = 1 if waveform.shape[1] <= waveform.shape[0] else 0
        return waveform.mean(axis=channel_axis).astype(np.float32)

    @staticmethod
    def _resample(mono: np.ndarray, sr: int) -> np.ndarray:
        """Resample mono audio to TARGET_SR (soxr, fallback: naive)."""
        if sr == TARGET_SR:
            return mono
        if _HAS_SOXR:
            return soxr.resample(mono, sr, TARGET_SR).astype(np.float32)
        # Fallback without soxr: integer decimation or linear interpolation.
        if sr % TARGET_SR == 0:
            return np.ascontiguousarray(mono[:: sr // TARGET_SR])
        n_out = int(round(len(mono) * TARGET_SR / sr))
        x_out = np.linspace(0.0, len(mono) - 1, n_out)
        return np.interp(x_out, np.arange(len(mono)), mono).astype(np.float32)

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
