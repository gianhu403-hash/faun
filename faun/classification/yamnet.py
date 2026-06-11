"""YAMNetAdapter — species classifier over YAMNet *base* embeddings.

Implements the frozen ``SpeciesClassifier`` protocol (see
``faun/classification/__init__.py``).

This adapter reuses the TF-Hub YAMNet model loaded by ``faun.ml.yamnet`` but
deliberately ignores the project's *anthropogenic* head (chainsaw / gunshot /
engine / …) — that head is useless for bird species. Instead it uses YAMNet's
1024-dim base embeddings as a feature vector and runs them through an optional
*probe* head (a small Keras or scikit-learn classifier) supplied by the caller.

Probe resolution order:
    1. explicit ``probe`` object passed to the constructor, or
    2. ``probe_path`` (constructor or ``YAMNET_PROBE_PATH`` env), loaded lazily.

If no probe is configured, ``classify`` returns a single
``Prediction("embedding_only", 0.0)`` and stashes the pooled embedding on
``self.last_embedding`` (also returned by ``embed``) so downstream code (e.g. a
few-shot k-NN) can use it.

TensorFlow / tensorflow_hub are imported lazily through ``faun.ml.yamnet``;
importing this module does not pull them.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import soxr

from faun.classification import Prediction

logger = logging.getLogger(__name__)

YAMNET_SR = 16_000
EMBEDDING_ONLY = "embedding_only"


class YAMNetAdapter:
    """YAMNet-embedding species classifier with an optional probe head.

    Args:
        probe: A pre-loaded probe model exposing ``predict_proba`` (sklearn) or
            ``__call__`` returning per-class scores (Keras). Takes precedence
            over ``probe_path``.
        probe_path: Path to a ``.keras`` or pickled sklearn probe. Falls back to
            the ``YAMNET_PROBE_PATH`` environment variable.
        labels: Optional class label list aligned with probe outputs. When
            absent, predictions are named ``class_<i>``.
        top_k: Maximum number of predictions returned by ``classify``.
    """

    def __init__(
        self,
        probe=None,
        probe_path: str | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self._probe = probe
        self.probe_path = probe_path or os.environ.get("YAMNET_PROBE_PATH")
        self.labels = labels
        self.top_k = top_k
        self.last_embedding: np.ndarray | None = None

    def _load_probe(self):
        """Lazily load the probe from ``probe_path`` (if any)."""
        if self._probe is not None:
            return self._probe
        if not self.probe_path:
            return None
        if self.probe_path.endswith(".keras") or self.probe_path.endswith(".h5"):
            import tensorflow as tf

            self._probe = tf.keras.models.load_model(self.probe_path)
        else:
            # The probe is an operator-supplied model file referenced by an
            # explicit local path / env var (never untrusted network input), so
            # pickle is acceptable here for sklearn-style probes.
            import pickle  # noqa: S403

            with open(self.probe_path, "rb") as fh:
                self._probe = pickle.load(fh)  # noqa: S301
        return self._probe

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return the mean-pooled 1024-dim YAMNet base embedding.

        Resamples to 16 kHz mono via ``soxr`` if needed, then runs the TF-Hub
        YAMNet model (loaded lazily through ``faun.ml.yamnet._load_models``).
        """
        from faun.ml import yamnet as ml_yamnet

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != YAMNET_SR:
            wav = soxr.resample(wav, sr, YAMNET_SR)

        model, _ = ml_yamnet._load_models()
        _scores, embeddings, _spec = model(wav)
        emb_np = np.asarray(embeddings.numpy())
        pooled = emb_np.mean(axis=0)
        self.last_embedding = pooled
        return pooled

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked species predictions from the probe head.

        With no probe configured, returns ``[Prediction("embedding_only", 0.0)]``
        and leaves the pooled embedding on ``self.last_embedding``.
        """
        embedding = self.embed(segment, sr)
        probe = self._load_probe()
        if probe is None:
            return [Prediction(EMBEDDING_ONLY, 0.0)]

        x = embedding[np.newaxis, :]
        if hasattr(probe, "predict_proba"):
            scores = np.asarray(probe.predict_proba(x))[0]
        else:
            out = probe(x)
            scores = np.asarray(out.numpy() if hasattr(out, "numpy") else out)[0]

        labels = self.labels or [f"class_{i}" for i in range(len(scores))]
        order = np.argsort(scores)[::-1][: self.top_k]
        return [Prediction(labels[i], float(scores[i])) for i in order]
