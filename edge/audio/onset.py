"""Sharp sound onset detector for ForestGuard.

Monitors audio stream and triggers ONLY when a sudden energy spike
(transient) is detected — chainsaw start, gunshot, axe hit, etc.

Algorithm:
1. Compute short-term energy in sliding windows
2. Compare against adaptive long-term energy threshold
3. Trigger when ratio exceeds configured threshold
4. Apply cooldown to prevent duplicate triggers
"""

import numpy as np
from dataclasses import dataclass

# Detection parameters
FRAME_SIZE = 512          # ~32ms at 16kHz — short-term analysis window
HOP_SIZE = 256            # ~16ms hop between frames
ENERGY_RATIO_THRESHOLD = 8.0   # short-term / long-term energy ratio to trigger
LONG_TERM_FRAMES = 30     # ~480ms of history for background energy estimate
MIN_ABSOLUTE_ENERGY = 1e-4     # ignore extremely quiet signals (silence floor)
COOLDOWN_FRAMES = 60      # ~960ms cooldown after trigger to avoid double fires


@dataclass
class OnsetEvent:
    """Describes a detected sharp sound onset."""
    triggered: bool
    frame_index: int       # which frame triggered
    energy_ratio: float    # peak ratio that caused trigger
    peak_energy: float     # absolute energy at trigger point


class OnsetDetector:
    """Stateful onset detector that maintains a rolling energy baseline.

    Usage:
        detector = OnsetDetector()
        event = detector.detect(waveform, sample_rate=16000)
        if event.triggered:
            # sharp sound detected — proceed to classify & triangulate
    """

    def __init__(
        self,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
        energy_ratio_threshold: float = ENERGY_RATIO_THRESHOLD,
        long_term_frames: int = LONG_TERM_FRAMES,
        min_absolute_energy: float = MIN_ABSOLUTE_ENERGY,
        cooldown_frames: int = COOLDOWN_FRAMES,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.energy_ratio_threshold = energy_ratio_threshold
        self.long_term_frames = long_term_frames
        self.min_absolute_energy = min_absolute_energy
        self.cooldown_frames = cooldown_frames

        # Rolling state
        self._energy_history: list[float] = []
        self._cooldown_counter = 0

    def reset(self) -> None:
        """Clear internal state (e.g., between recording sessions)."""
        self._energy_history.clear()
        self._cooldown_counter = 0

    def detect(self, waveform: np.ndarray, sample_rate: int = 16000) -> OnsetEvent:
        """Analyze a waveform chunk for sharp sound onsets.

        Args:
            waveform: 1-D float32 array of audio samples.
            sample_rate: Sample rate (used for future extensions).

        Returns:
            OnsetEvent with triggered=True if a sharp transient was found.
        """
        if len(waveform) < self.frame_size:
            return OnsetEvent(triggered=False, frame_index=0, energy_ratio=0.0, peak_energy=0.0)

        # Compute per-frame RMS energy
        n_frames = (len(waveform) - self.frame_size) // self.hop_size + 1
        energies = np.empty(n_frames, dtype=np.float64)

        for i in range(n_frames):
            start = i * self.hop_size
            frame = waveform[start : start + self.frame_size].astype(np.float64)
            energies[i] = np.sqrt(np.mean(frame ** 2))  # RMS

        # Scan frames for onset
        best_ratio = 0.0
        best_frame = 0
        best_energy = 0.0
        triggered = False

        for i in range(n_frames):
            e = energies[i]
            self._energy_history.append(e)

            # Maintain rolling window
            if len(self._energy_history) > self.long_term_frames + self.cooldown_frames:
                self._energy_history.pop(0)

            # Cooldown check
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1
                continue

            # Need enough history for a baseline
            if len(self._energy_history) < self.long_term_frames:
                continue

            # Long-term baseline: median of recent history (robust to outliers)
            baseline_window = self._energy_history[-(self.long_term_frames + 1) : -1]
            if not baseline_window:
                continue
            baseline = float(np.median(baseline_window))

            # Skip if signal is below absolute floor
            if e < self.min_absolute_energy:
                continue

            # Compute ratio
            if baseline < 1e-10:
                # Baseline is near silence — any non-trivial signal is a spike
                ratio = e / self.min_absolute_energy
            else:
                ratio = e / baseline

            if ratio > best_ratio:
                best_ratio = ratio
                best_frame = i
                best_energy = e

            if ratio >= self.energy_ratio_threshold and not triggered:
                triggered = True
                self._cooldown_counter = self.cooldown_frames

        return OnsetEvent(
            triggered=triggered,
            frame_index=best_frame,
            energy_ratio=best_ratio,
            peak_energy=best_energy,
        )


def detect_onset(waveform: np.ndarray, sample_rate: int = 16000) -> OnsetEvent:
    """Stateless convenience function: detect sharp sound in a single chunk."""
    detector = OnsetDetector()
    # Pre-fill with a quiet baseline so the detector has context
    quiet_baseline = np.zeros(FRAME_SIZE * LONG_TERM_FRAMES, dtype=np.float32)
    detector.detect(quiet_baseline, sample_rate)
    return detector.detect(waveform, sample_rate)
