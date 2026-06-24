"""Tests for faun.segmentation — SegmentExtractor (downmix + resample + onset)."""

from __future__ import annotations

import numpy as np
import pytest

from faun.segmentation import Segment, SegmentExtractor, TARGET_SR

SR_48K = 48000


# ---------------------------------------------------------------------------
# Synthetic WAV helpers (adapted from tests/test_onset.py generators, 48 kHz)
# ---------------------------------------------------------------------------


def _silence(duration_s: float, sr: int = SR_48K) -> np.ndarray:
    """Near-zero noise floor."""
    rng = np.random.default_rng(7)
    return rng.normal(0, 1e-5, int(sr * duration_s)).astype(np.float32)


def _burst(
    duration_s: float = 1.0, sr: int = SR_48K, amplitude: float = 0.7
) -> np.ndarray:
    """Loud sine burst simulating a sharp sound (chainsaw-like)."""
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)


def _to_stereo(mono: np.ndarray) -> np.ndarray:
    """Duplicate mono into (frames, 2) stereo, like soundfile.read returns."""
    return np.column_stack([mono, mono])


def _silence_burst_silence(
    pre_s: float = 2.0, burst_s: float = 1.0, post_s: float = 2.0, sr: int = SR_48K
) -> np.ndarray:
    return np.concatenate(
        [_silence(pre_s, sr), _burst(burst_s, sr), _silence(post_s, sr)]
    )


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------


class TestSegment:
    def test_fields_and_end(self) -> None:
        seg = Segment(start_s=1.5, duration_s=3.0)
        assert seg.start_s == 1.5
        assert seg.duration_s == 3.0
        assert seg.end_s == 4.5


# ---------------------------------------------------------------------------
# Detection on synthetic 48 kHz stereo
# ---------------------------------------------------------------------------


class TestSegmentExtractor:
    def test_burst_detected_stereo_48k(self) -> None:
        """Silence -> burst -> silence (48 kHz stereo) yields a segment near 2.0s."""
        waveform = _to_stereo(_silence_burst_silence(pre_s=2.0))
        segments = SegmentExtractor().extract(waveform, SR_48K)
        assert len(segments) >= 1
        seg = segments[0]
        # Onset is at 2.0s; allow padding + frame granularity tolerance.
        assert seg.start_s == pytest.approx(2.0, abs=0.5)
        assert seg.duration_s >= 0.5

    def test_silence_gives_zero_segments(self) -> None:
        waveform = _to_stereo(_silence(5.0))
        assert SegmentExtractor().extract(waveform, SR_48K) == []

    def test_mono_input_accepted(self) -> None:
        waveform = _silence_burst_silence(pre_s=2.0)
        segments = SegmentExtractor().extract(waveform, SR_48K)
        assert len(segments) >= 1

    def test_two_separated_bursts_give_two_segments(self) -> None:
        """Two bursts >max_segment apart must produce two distinct segments."""
        sr = SR_48K
        waveform = np.concatenate(
            [
                _silence(2.0, sr),
                _burst(0.8, sr),
                _silence(8.0, sr),
                _burst(0.8, sr),
                _silence(2.0, sr),
            ]
        )
        segments = SegmentExtractor().extract(_to_stereo(waveform), sr)
        assert len(segments) == 2
        assert segments[0].start_s == pytest.approx(2.0, abs=0.5)
        # Second burst starts at 2.0 + 0.8 + 8.0 = 10.8s.
        assert segments[1].start_s == pytest.approx(10.8, abs=0.5)
        assert segments[1].start_s > segments[0].end_s

    def test_segments_within_file_bounds(self) -> None:
        waveform = _silence_burst_silence(pre_s=1.0, burst_s=0.8, post_s=0.5)
        total_s = len(waveform) / SR_48K
        for seg in SegmentExtractor().extract(_to_stereo(waveform), SR_48K):
            assert seg.start_s >= 0.0
            assert seg.end_s <= total_s + 1e-6

    def test_max_segment_cap_respected(self) -> None:
        waveform = _to_stereo(
            _silence_burst_silence(pre_s=2.0, burst_s=3.0, post_s=3.0)
        )
        extractor = SegmentExtractor(max_segment_s=2.0)
        segments = extractor.extract(waveform, SR_48K)
        assert segments
        assert all(seg.duration_s <= 2.0 + 1e-6 for seg in segments)

    def test_native_16k_passthrough(self) -> None:
        """Already-16k mono audio is processed without resampling."""
        waveform = _silence_burst_silence(pre_s=2.0, sr=TARGET_SR)
        segments = SegmentExtractor().extract(waveform, TARGET_SR)
        assert len(segments) >= 1
        assert segments[0].start_s == pytest.approx(2.0, abs=0.5)

    def test_empty_waveform(self) -> None:
        assert SegmentExtractor().extract(np.array([], dtype=np.float32), SR_48K) == []

    def test_invalid_sr_raises(self) -> None:
        with pytest.raises(ValueError):
            SegmentExtractor().extract(_silence(1.0), 0)

    def test_invalid_constructor_params_raise(self) -> None:
        with pytest.raises(ValueError):
            SegmentExtractor(min_segment_s=5.0, max_segment_s=1.0)
        with pytest.raises(ValueError):
            SegmentExtractor(padding_s=-1.0)


# ---------------------------------------------------------------------------
# Honest-segmentation knobs (ADR-0006): dense windows, silence filter, NMS.
# Every one OFF by default — SC-B1 (golden) lives in test_golden.py; here we
# assert the ON behaviour and that the default path is untouched.
# ---------------------------------------------------------------------------


class TestHonestSegmentation:
    def _two_bursts_6s(self) -> np.ndarray:
        """6 s mono 48k: quiet floor + 0.5 s bursts at 1.0 s and 4.0 s."""
        sr = SR_48K
        t = np.arange(int(sr * 6)) / sr
        sig = (1e-5 * np.random.default_rng(7).standard_normal(t.size)).astype(
            np.float32
        )
        for centre in (1.0, 4.0):
            m = (t >= centre) & (t < centre + 0.5)
            sig[m] += (0.7 * np.sin(2 * np.pi * 500.0 * t[m])).astype(np.float32)
        return sig

    def test_default_path_is_onset_not_grid(self) -> None:
        """All knobs OFF -> identical to the bare onset extractor (no grid)."""
        wav = self._two_bursts_6s()
        base = SegmentExtractor().extract(wav, SR_48K)
        same = SegmentExtractor(
            dense_windows=False, silence_filter=False, nms_iou=None
        ).extract(wav, SR_48K)
        assert base == same

    def test_dense_windows_tile_the_recording(self) -> None:
        """5 s windows @ 2.5 s hop over 6 s -> starts 0.0/2.5/5.0 (SC-B3)."""
        wav = self._two_bursts_6s()
        segs = SegmentExtractor(dense_windows=True).extract(wav, SR_48K)
        assert [round(s.start_s, 2) for s in segs] == [0.0, 2.5, 5.0]
        assert round(segs[0].duration_s, 2) == 5.0  # full window
        assert round(segs[-1].duration_s, 2) == 1.0  # clamped to file end

    def test_dense_window_geometry_is_configurable(self) -> None:
        wav = self._two_bursts_6s()
        segs = SegmentExtractor(dense_windows=True, window_s=2.0, hop_s=2.0).extract(
            wav, SR_48K
        )
        assert [round(s.start_s, 2) for s in segs] == [0.0, 2.0, 4.0]
        assert all(round(s.duration_s, 2) == 2.0 for s in segs)

    def test_silence_filter_drops_empty_windows(self) -> None:
        """Dense grid over two bursts: the all-silence tail window is pruned."""
        wav = self._two_bursts_6s()
        kept = SegmentExtractor(dense_windows=True, silence_filter=True).extract(
            wav, SR_48K
        )
        starts = [round(s.start_s, 2) for s in kept]
        # The 5.0–6.0 window holds only noise floor -> dropped; 0.0 and 2.5 keep
        # a burst (1.0 s resp. 4.0 s) -> retained.
        assert 5.0 not in starts
        assert 0.0 in starts and 2.5 in starts

    def test_nms_collapses_overlapping_windows(self) -> None:
        wav = self._two_bursts_6s()
        dense = SegmentExtractor(dense_windows=True).extract(wav, SR_48K)
        nmsed = SegmentExtractor(dense_windows=True, nms_iou=0.3).extract(wav, SR_48K)
        assert len(nmsed) < len(dense)
        # Greedy keep-earliest: the 0.0 window survives, the heavily-overlapping
        # 2.5 window is suppressed (IoU 0–5 vs 2.5–6 = 2.5/6 ≈ 0.42 > 0.3).
        starts = [round(s.start_s, 2) for s in nmsed]
        assert starts[0] == 0.0
        assert 2.5 not in starts

    def test_nms_keeps_disjoint_windows(self) -> None:
        """Non-overlapping windows survive any IoU threshold."""
        wav = self._two_bursts_6s()
        segs = SegmentExtractor(
            dense_windows=True, window_s=2.0, hop_s=2.0, nms_iou=0.1
        ).extract(wav, SR_48K)
        # 0–2, 2–4, 4–6 are disjoint (IoU 0) -> all kept.
        assert [round(s.start_s, 2) for s in segs] == [0.0, 2.0, 4.0]

    def test_nms_threshold_is_strict_keep_on_equality(self) -> None:
        """Suppression is strict (IoU > threshold), so IoU == threshold is kept.

        2 s windows @ 1 s hop -> adjacent windows overlap 1 s, IoU = 1/3. With
        nms_iou exactly 1/3 the adjacent pair is on the boundary and retained;
        a hair below it suppresses.
        """
        wav = self._two_bursts_6s()
        kept_eq = SegmentExtractor(
            dense_windows=True, window_s=2.0, hop_s=1.0, nms_iou=1.0 / 3.0
        ).extract(wav, SR_48K)
        kept_lo = SegmentExtractor(
            dense_windows=True, window_s=2.0, hop_s=1.0, nms_iou=0.33
        ).extract(wav, SR_48K)
        assert len(kept_eq) > len(kept_lo)  # boundary kept vs just-below suppressed

    def test_invalid_new_params_raise(self) -> None:
        with pytest.raises(ValueError):
            SegmentExtractor(window_s=0.0)
        with pytest.raises(ValueError):
            SegmentExtractor(hop_s=-1.0)
        with pytest.raises(ValueError):
            SegmentExtractor(nms_iou=0.0)
        with pytest.raises(ValueError):
            SegmentExtractor(nms_iou=1.5)


class TestDownmixResample:
    def test_downmix_is_channel_mean(self) -> None:
        left = np.full(1000, 0.4, dtype=np.float32)
        right = np.full(1000, 0.2, dtype=np.float32)
        mono = SegmentExtractor._downmix(np.column_stack([left, right]))
        assert mono.shape == (1000,)
        assert mono == pytest.approx(np.full(1000, 0.3), abs=1e-6)

    def test_channels_first_layout_accepted(self) -> None:
        stereo_cf = np.stack([np.zeros(1000), np.ones(1000)]).astype(np.float32)
        mono = SegmentExtractor._downmix(stereo_cf)
        assert mono.shape == (1000,)
        assert mono == pytest.approx(np.full(1000, 0.5), abs=1e-6)

    def test_resample_48k_halves_length_to_16k_third(self) -> None:
        mono = _burst(1.0, SR_48K)
        out = SegmentExtractor._resample(mono, SR_48K)
        assert len(out) == pytest.approx(TARGET_SR, abs=2)

    def test_decimation_fallback_without_soxr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The soxr-vs-fallback choice now lives in faun.audio (ADR-0002), so the
        # fallback is exercised by patching the flag there.
        import faun.audio as audio_mod

        monkeypatch.setattr(audio_mod, "_HAS_SOXR", False)
        mono = _burst(1.0, SR_48K)
        out = SegmentExtractor._resample(mono, SR_48K)
        assert len(out) == TARGET_SR
        # Burst must still be detectable end-to-end through the fallback.
        waveform = _silence_burst_silence(pre_s=2.0)
        segments = SegmentExtractor().extract(waveform, SR_48K)
        assert len(segments) >= 1
