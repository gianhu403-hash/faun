"""PerchProbeAdapter — species classifier over **Perch 2** embeddings + a probe.

Implements the frozen ``SpeciesClassifier`` protocol (see
``faun/classification/__init__.py``).

This is the *served* counterpart of the retraining loop: where
:class:`~faun.classification.perch_v2.Perch2Adapter` returns Perch 2's own
~14.8k-class zero-shot logits, this adapter feeds Perch 2's **1536-dim
embedding** into a small *probe* head trained on our own ground-truth labels
(``faun.retraining.train_probe_cv`` / ``faun finetune``-free, CPU). It mirrors
:class:`~faun.classification.yamnet.YAMNetAdapter` (embedding + optional probe)
but on the Perch 2 backbone instead of YAMNet.

Probe resolution order:
    1. explicit ``probe`` object passed to the constructor, or
    2. ``probe_path`` (constructor or ``PERCH_V2_PROBE_PATH`` via
       :func:`faun.settings.get_settings`), loaded lazily.

If no probe is configured, ``classify`` returns a single
``Prediction("embedding_only", 0.0)`` and stashes the pooled embedding on
``self.last_embedding`` (also returned by ``embed``) so downstream code (e.g. a
few-shot k-NN) can use it.

TensorFlow is imported lazily through :class:`Perch2Adapter` (its ``embed`` pulls
TF only at call time); importing this module pulls neither TF nor kagglehub.
"""

from __future__ import annotations

import logging

import numpy as np

from faun.classification import Prediction
from faun.settings import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_ONLY = "embedding_only"


class PerchProbeAdapter:
    """Perch 2-embedding species classifier with a trained probe head.

    Args:
        probe: A pre-loaded probe exposing ``predict_proba`` (sklearn) or
            ``__call__`` returning per-class scores (Keras). Takes precedence
            over ``probe_path``.
        probe_path: Path to a ``.keras``/``.h5`` or pickled sklearn probe. Falls
            back to ``PERCH_V2_PROBE_PATH`` (via ``get_settings``).
        model_path: Optional Perch 2 SavedModel path forwarded to the embedder
            (else the embedder resolves ``PERCH_V2_MODEL_PATH`` / kagglehub).
        labels: Optional class label list aligned with the probe outputs. When
            absent, names come from the probe's ``classes_`` (sklearn) and only
            then fall back to ``species_<i>``.
        top_k: Maximum number of predictions returned by ``classify``.
    """

    def __init__(
        self,
        probe=None,
        probe_path: str | None = None,
        model_path: str | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self._probe = probe
        self.probe_path = probe_path or get_settings().perch_v2_probe_path
        self.model_path = model_path
        self.labels = labels
        self.top_k = top_k
        self._embedder = None
        self.last_embedding: np.ndarray | None = None

    def _get_embedder(self):
        """Lazily build the Perch 2 embedder (it pulls TF only at call time)."""
        if self._embedder is None:
            from faun.classification.perch_v2 import Perch2Adapter

            self._embedder = Perch2Adapter(model_path=self.model_path)
        return self._embedder

    def _load_probe(self):
        """Lazily load the probe from ``probe_path`` (if any)."""
        if self._probe is not None:
            return self._probe
        if not self.probe_path:
            return None
        if self.probe_path.endswith((".keras", ".h5")):
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
        """Return the Perch 2 1536-dim embedding (delegated to Perch2Adapter)."""
        embedding = np.asarray(self._get_embedder().embed(waveform, sr))
        self.last_embedding = embedding
        return embedding

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked species predictions from the probe head.

        With no probe configured, returns ``[Prediction("embedding_only", 0.0)]``
        and leaves the pooled embedding on ``self.last_embedding``.

        Raises:
            RuntimeError: if TensorFlow is unavailable at call time (via the
                Perch 2 embedder).
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

        # Probe column i corresponds to classes_[i] (sklearn) — use the trained
        # species names; fall back to an explicit labels arg, then species_<i>.
        classes = getattr(probe, "classes_", None)
        order = np.argsort(scores)[::-1][: self.top_k]

        def _name(i: int) -> str:
            if self.labels is not None and i < len(self.labels):
                return self.labels[i]
            if classes is not None and i < len(classes):
                return str(classes[i])
            return f"species_{i}"

        return [Prediction(_name(int(i)), float(scores[i])) for i in order]
