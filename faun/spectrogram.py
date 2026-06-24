"""Spectrogram PNG rendering for the review UI (FR-008, ADR-0009).

A self-contained copy of ``experiments.common.save_spectrogram_png`` (+ the tiny
framing helper it relied on) so the pipeline package NEVER imports from
``experiments/`` — that tree is benchmark scaffolding, not a runtime dependency.

``matplotlib`` is imported LAZILY inside :func:`save_spectrogram_png` (Agg
backend, headless) so importing this module costs nothing and stays TF/heavy-dep
free; the dependency itself is pinned in ``requirements-pipeline.txt`` so the
deploy image has it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["save_spectrogram_png"]


def _frame(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice a 1-D signal into ``[n_win, win]`` frames (tail dropped).

    A signal shorter than one window is zero-padded to exactly one window —
    mirrors ``experiments.common.windows`` so the spectrogram matches the
    benchmark renderer.
    """
    x = np.asarray(x)
    if win <= 0 or hop <= 0:
        raise ValueError(f"win/hop must be positive, got win={win} hop={hop}")
    if len(x) < win:
        return np.pad(x, (0, win - len(x)))[None, :]
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def save_spectrogram_png(
    x: np.ndarray,
    sr: int,
    out_path: str | Path,
    *,
    n_fft: int = 1024,
    hop: int = 256,
) -> Path:
    """Render a log-STFT spectrogram of ``x`` to ``out_path`` as a PNG.

    Mono is assumed; a multi-channel clip is downmixed by channel mean first.
    matplotlib (Agg) is imported lazily here so the module import stays cheap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:  # downmix stereo/multi to mono for the display
        x = x.mean(axis=1)

    frames = _frame(x, n_fft, hop)
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)).T
    log_spec = 20.0 * np.log10(spec + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [0, len(x) / sr if sr else 0, 0, (sr / 2 / 1000) if sr else 0]
    ax.imshow(log_spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("kHz")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
