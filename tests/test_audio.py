"""Tests for faun.audio — the single owner of audio preprocessing (ADR-0002).

Covers downmix / resample / fit_window directly (TF/torch-free), the soxr-less
fallback path, and the frozen re-export invariant that faun.embeddings and
faun.training.dataset rely on (faun.embeddings._downmix is faun.audio.downmix,
etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from faun import audio


# ---------------------------------------------------------------------------
# downmix
# ---------------------------------------------------------------------------


class TestDownmix:
    def test_mono_passthrough(self) -> None:
        mono = np.arange(100, dtype=np.float32)
        out = audio.downmix(mono)
        assert out.shape == (100,)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, mono)

    def test_stereo_frames_channels_mean(self) -> None:
        # (frames, channels): more frames than channels.
        left = np.full(1000, 0.4, dtype=np.float32)
        right = np.full(1000, 0.2, dtype=np.float32)
        out = audio.downmix(np.column_stack([left, right]))
        assert out.shape == (1000,)
        assert out == pytest.approx(np.full(1000, 0.3), abs=1e-6)

    def test_channels_frames_heuristic(self) -> None:
        # (channels, frames): fewer rows than columns -> mean over axis 0.
        stereo_cf = np.stack([np.zeros(1000), np.ones(1000)]).astype(np.float32)
        out = audio.downmix(stereo_cf)
        assert out.shape == (1000,)
        assert out == pytest.approx(np.full(1000, 0.5), abs=1e-6)

    def test_ndim_error(self) -> None:
        with pytest.raises(ValueError):
            audio.downmix(np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_sr_passthrough_float32(self) -> None:
        mono = np.arange(10, dtype=np.float32)
        out = audio.resample(mono, 16000, 16000)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, mono)

    def test_soxr_path_length_approx_ratio(self) -> None:
        # 48k -> 16k should shrink by ~1/3 (soxr is installed locally).
        mono = np.sin(np.linspace(0, 50, 48000)).astype(np.float32)
        out = audio.resample(mono, 48000, 16000)
        assert out.dtype == np.float32
        assert len(out) == pytest.approx(16000, abs=2)

    def test_nonpositive_sr_raises(self) -> None:
        with pytest.raises(ValueError):
            audio.resample(np.zeros(10, dtype=np.float32), 0, 16000)
        with pytest.raises(ValueError):
            audio.resample(np.zeros(10, dtype=np.float32), -1, 16000)

    def test_fallback_without_soxr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linear-interp fallback when soxr is unavailable (ADR-0002 owner)."""
        monkeypatch.setattr(audio, "_HAS_SOXR", False)
        mono = np.sin(np.linspace(0, 50, 48000)).astype(np.float32)
        out = audio.resample(mono, 48000, 16000)
        assert out.dtype == np.float32
        # round(48000 * 16000 / 48000) == 16000 exactly.
        assert len(out) == 16000


# ---------------------------------------------------------------------------
# fit_window
# ---------------------------------------------------------------------------


class TestFitWindow:
    def test_pad_right(self) -> None:
        mono = np.ones(10, dtype=np.float32)
        out = audio.fit_window(mono, 16)
        assert out.shape == (16,)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out[:10], mono)
        np.testing.assert_array_equal(out[10:], np.zeros(6, dtype=np.float32))

    def test_truncate(self) -> None:
        mono = np.arange(20, dtype=np.float32)
        out = audio.fit_window(mono, 8)
        assert out.shape == (8,)
        np.testing.assert_array_equal(out, mono[:8])

    def test_exact(self) -> None:
        mono = np.arange(8, dtype=np.float32)
        out = audio.fit_window(mono, 8)
        assert out.shape == (8,)
        np.testing.assert_array_equal(out, mono)


# ---------------------------------------------------------------------------
# Frozen re-export invariant (faun.training.dataset depends on these names)
# ---------------------------------------------------------------------------


def test_embeddings_reexports_are_identical_objects() -> None:
    import faun.audio
    import faun.embeddings

    assert faun.embeddings._downmix is faun.audio.downmix
    assert faun.embeddings._resample is faun.audio.resample
    assert faun.embeddings._fit_window is faun.audio.fit_window
