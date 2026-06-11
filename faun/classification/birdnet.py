"""BirdNETAdapter — bird species classifier backed by ``birdnetlib``.

Implements the frozen ``SpeciesClassifier`` protocol (see
``faun/classification/__init__.py``).

Heavy dependencies (``birdnetlib`` + the bundled TensorFlow model) are NOT in
``requirements-pipeline.txt`` and are imported lazily *inside* methods. Importing
this module therefore never pulls TensorFlow (Lesson 11). If ``birdnetlib`` is
unavailable at call time, ``classify`` raises ``RuntimeError`` with an
actionable message rather than silently returning an empty list.

BirdNET expects 48 kHz mono 3-second windows; ``classify`` resamples the input
to 48 kHz mono via ``soxr`` before handing it to the analyzer.

License note: BirdNET is CC BY-NC-SA (non-commercial + ShareAlike).
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import soundfile as sf
import soxr

from faun.classification import Prediction

logger = logging.getLogger(__name__)

BIRDNET_SR = 48_000


class BirdNETAdapter:
    """BirdNET species classifier.

    Args:
        lat: Optional latitude for location-aware species filtering.
        lon: Optional longitude for location-aware species filtering.
        date: Optional ``datetime.date`` for season-aware filtering.
        min_conf: Minimum confidence reported by birdnetlib.
        top_k: Maximum number of predictions returned by ``classify``.
    """

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        date=None,
        min_conf: float = 0.1,
        top_k: int = 5,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.date = date
        self.min_conf = min_conf
        self.top_k = top_k
        self._analyzer = None

    def _load(self):
        """Lazily construct the birdnetlib Analyzer.

        Raises:
            RuntimeError: if ``birdnetlib`` is not installed.
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            from birdnetlib.analyzer import Analyzer
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise RuntimeError(
                "BirdNETAdapter requires the 'birdnetlib' package, which is not "
                "installed. Install it with `pip install birdnetlib` (pulls "
                "TensorFlow + the BirdNET model). It is intentionally excluded "
                "from requirements-pipeline.txt."
            ) from exc
        self._analyzer = Analyzer()
        return self._analyzer

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked bird-species predictions for ``segment``.

        Args:
            segment: 1-D (mono) or 2-D (multi-channel) waveform.
            sr: Sample rate of ``segment`` in Hz.

        Returns:
            Up to ``top_k`` ``Prediction`` items sorted by descending
            probability.

        Raises:
            RuntimeError: if ``birdnetlib`` is unavailable.
        """
        analyzer = self._load()

        waveform = np.asarray(segment, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != BIRDNET_SR:
            waveform = soxr.resample(waveform, sr, BIRDNET_SR)

        from birdnetlib import Recording

        # birdnetlib reads from disk; write the resampled mono signal to a temp
        # WAV and feed it through a Recording.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, waveform, BIRDNET_SR)
            kwargs: dict = {"min_conf": self.min_conf}
            if self.lat is not None and self.lon is not None:
                kwargs["lat"] = self.lat
                kwargs["lon"] = self.lon
            if self.date is not None:
                kwargs["date"] = self.date
            recording = Recording(analyzer, tmp_path, **kwargs)
            recording.analyze()
            detections = recording.detections
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning("Failed to remove temp file %s: %s", tmp_path, exc)

        # Aggregate the best confidence per species across all 3-s windows.
        best: dict[str, float] = {}
        for det in detections:
            species = det.get("scientific_name") or det.get("common_name") or "unknown"
            conf = float(det.get("confidence", 0.0))
            if conf > best.get(species, -1.0):
                best[species] = conf

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [Prediction(sp, conf) for sp, conf in ranked[: self.top_k]]
