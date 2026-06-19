"""Unit tests for faun.pipeline — the shared segment->classify->Detection core.

TF/torch-free: drives run_batch with the StubAdapter via build_labels. The key
contract under test is the clip<->detection ROW-ALIGNMENT (ADR-0003) that
batch_label's embeddings export depends on, plus slice_clip / to_classifier_input.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from faun.classification import StubAdapter
from faun.detections import SOURCE_STUB, STATUS_PSEUDO, Label
from faun.pipeline import (
    CLASSIFY_SR,
    SegmentResult,
    run_batch,
    slice_clip,
    to_classifier_input,
)
from faun.segmentation import Segment

SR = 48_000


def _burst_wav(path: Path) -> None:
    """A 6 s stereo 48 kHz WAV with a loud 3 kHz burst at ~2.5–3.0 s (one onset)."""
    t = np.linspace(0, 6, 6 * SR, endpoint=False)
    sig = 0.005 * np.random.default_rng(7).standard_normal((6 * SR, 2))
    burst = np.sin(2 * np.pi * 3000 * t) * ((t > 2.5) & (t < 3.0))
    sig[:, 0] += burst
    sig[:, 1] += burst
    sf.write(path, sig, SR)


def _read_f64(path):
    return sf.read(path, dtype="float64", always_2d=False)


def _stub_labels(clip_16k: np.ndarray) -> list[Label]:
    return [
        Label.from_prediction(p, source=SOURCE_STUB, status=STATUS_PSEUDO)
        for p in StubAdapter().classify(clip_16k, CLASSIFY_SR)
    ]


def test_run_batch_aligns_clips_with_detections(tmp_path: Path) -> None:
    wav = tmp_path / "REC.wav"
    _burst_wav(wav)
    entry = SimpleNamespace(path=wav, trap_id="A1")

    results = list(
        run_batch([entry], read_waveform=_read_f64, build_labels=_stub_labels)
    )

    assert results, "the burst must yield >=1 SegmentResult"
    for r in results:
        assert isinstance(r, SegmentResult)
        # The clip is the ORIGINAL sr + channels (preserved for disk/embedding).
        assert r.sr == SR
        assert r.clip.dtype == np.float64
        assert r.clip.ndim == 2  # stereo preserved
        # Detection metadata is wired from the entry; labels came from the stub.
        assert r.detection.trap_id == "A1"
        assert r.detection.source_file == "REC.wav"
        assert {lbl.species for lbl in r.detection.labels} >= {"Turdus merula"}
        # clip length is consistent with the segment duration (alignment sanity).
        assert abs(len(r.clip) / r.sr - r.detection.segment.duration_s) < 0.05
        # segment_path basename is derived from the detection id (clip on disk).
        assert r.detection.segment_path == f"segments/{r.detection.detection_id}.wav"


def test_run_batch_empty_entries_yields_nothing() -> None:
    assert list(run_batch([], read_waveform=_read_f64, build_labels=_stub_labels)) == []


def test_to_classifier_input_is_mono_16k(tmp_path: Path) -> None:
    wav = tmp_path / "REC.wav"
    _burst_wav(wav)
    waveform, sr = _read_f64(wav)
    clip = slice_clip(waveform, sr, Segment(start_s=2.5, duration_s=0.5))
    assert clip.ndim == 2  # stereo clip on the original timeline

    x = to_classifier_input(clip, sr)
    assert x.ndim == 1  # downmixed to mono
    assert x.dtype == np.float32
    # 48k -> 16k: ~1/3 the samples (soxr length is exact-ish; allow a few samples).
    assert abs(len(x) - round(len(clip) * CLASSIFY_SR / sr)) <= 5


def test_slice_clip_clamps_out_of_bounds() -> None:
    waveform = np.arange(100, dtype=np.float64)
    # A segment running past the end clamps to the waveform length (no IndexError).
    clip = slice_clip(waveform, sr=10, segment=Segment(start_s=0.0, duration_s=1000.0))
    assert len(clip) == 100
    assert clip[0] == 0.0
