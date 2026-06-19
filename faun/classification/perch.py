"""PerchAdapter — bird species classifier backed by Google's Perch model.

Implements the frozen ``SpeciesClassifier`` protocol (see
``faun/classification/__init__.py``).

Perch is a TensorFlow SavedModel served from TF-Hub / Kaggle. Perch 1 (the
default here) is downloadable without authentication and exposes 1280-dim
embeddings plus per-species logits (~10k classes). It expects 32 kHz mono
5-second windows; ``classify``/``embed`` resample to 32 kHz mono and pad / crop
to exactly 5 s before inference.

TensorFlow + tensorflow_hub are NOT in ``requirements-pipeline.txt`` and are
imported lazily inside ``_load`` (Lesson 11). Importing this module never pulls
TensorFlow. If TF is unavailable at call time, ``classify``/``embed`` raise
``RuntimeError`` rather than failing silently.

Model source resolution:
    1. ``model_path`` constructor argument, or
    2. ``PERCH_MODEL_PATH`` (via :func:`faun.settings.get_settings`), or
    3. ``DEFAULT_TFHUB_URL`` (Perch 1, no auth).
"""

from __future__ import annotations

import logging

import numpy as np
import soxr

from faun.classification import Prediction
from faun.settings import get_settings

logger = logging.getLogger(__name__)

PERCH_SR = 32_000
PERCH_WINDOW_S = 5
PERCH_WINDOW_SAMPLES = PERCH_SR * PERCH_WINDOW_S  # 160_000
DEFAULT_TFHUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/1"


class PerchAdapter:
    """Perch (bird vocalization) species classifier.

    Args:
        model_path: Local SavedModel directory or TF-Hub URL. Falls back to
            ``PERCH_MODEL_PATH`` (via ``get_settings``), then
            ``DEFAULT_TFHUB_URL``.
        labels: Optional class label list aligned with the logits output. When
            absent, predictions are named ``species_<i>``.
        top_k: Maximum number of predictions returned by ``classify``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self.model_path = (
            model_path or get_settings().perch_model_path or DEFAULT_TFHUB_URL
        )
        self.labels = labels
        self.top_k = top_k
        self._model = None

    def _load(self):
        """Lazily load the Perch SavedModel via tensorflow_hub.

        Raises:
            RuntimeError: if TensorFlow / tensorflow_hub are not installed.
        """
        if self._model is not None:
            return self._model
        try:
            import tensorflow_hub as hub
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise RuntimeError(
                "PerchAdapter requires TensorFlow and 'tensorflow_hub', which "
                "are not installed. Install them with `pip install tensorflow "
                "tensorflow_hub`. They are intentionally excluded from "
                "requirements-pipeline.txt."
            ) from exc
        self._model = hub.load(self.model_path)
        return self._model

    def _prepare(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Resample to 32 kHz mono and pad / crop to exactly 5 s."""
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != PERCH_SR:
            wav = soxr.resample(wav, sr, PERCH_SR)
        if len(wav) < PERCH_WINDOW_SAMPLES:
            wav = np.pad(wav, (0, PERCH_WINDOW_SAMPLES - len(wav)))
        elif len(wav) > PERCH_WINDOW_SAMPLES:
            wav = wav[:PERCH_WINDOW_SAMPLES]
        return wav

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    def _infer(self, waveform: np.ndarray, sr: int):
        """Run the model, returning ``(logits, embedding)`` as numpy arrays."""
        model = self._load()
        wav = self._prepare(waveform, sr)
        out = model.infer_tf(wav[np.newaxis, :])
        # Perch returns a (logits, embeddings) tuple/dict depending on version.
        if isinstance(out, dict):
            logits = out.get("label") if "label" in out else out.get("logits")
            embedding = out.get("embedding")
        else:
            logits, embedding = out[0], out[1]
        return self._to_numpy(logits), self._to_numpy(embedding)

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return the Perch embedding for ``waveform`` (for few-shot k-NN).

        Raises:
            RuntimeError: if TensorFlow / tensorflow_hub are unavailable.
        """
        _logits, embedding = self._infer(waveform, sr)
        return embedding[0] if embedding.ndim > 1 else embedding

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked species predictions for ``segment``.

        Raises:
            RuntimeError: if TensorFlow / tensorflow_hub are unavailable.
        """
        logits, _embedding = self._infer(segment, sr)
        scores = logits[0] if logits.ndim > 1 else logits
        labels = self.labels or [f"species_{i}" for i in range(len(scores))]
        order = np.argsort(scores)[::-1][: self.top_k]
        return [Prediction(labels[i], float(scores[i])) for i in order]
