"""Perch2Adapter — bird species classifier backed by Google's **Perch 2**.

Implements the frozen ``SpeciesClassifier`` protocol (see
``faun/classification/__init__.py``).

Perch 2 (arXiv 2508.04665, Apache 2.0) is a TensorFlow SavedModel served from
Kaggle Models. Unlike Perch 1 it exposes **1536-dim** embeddings (``embedding``),
a 5x3 spatial embedding, and per-species logits (``label``, ~14.8k classes). It
keeps Perch 1's input geometry: 32 kHz mono, 5-second windows = 160000 samples
float32. ``classify`` / ``embed`` resample to 32 kHz mono, peak-normalize, and
pad / crop to exactly 5 s before inference.

Inference path (NOT Perch 1's ``hub.load().infer_tf``)::

    model = tf.saved_model.load(path)
    out = model.signatures["serving_default"](inputs=data)   # data: (N, 160000)
    # out -> {"embedding"(N,1536), "spatial_embedding"(N,5,3,1536),
    #         "label"(N,~14795), "spectrogram"}

Source resolution (in order):
    1. ``model_path`` constructor argument (local SavedModel dir), or
    2. ``get_settings().perch_v2_model_path`` (the ``PERCH_V2_MODEL_PATH`` env), or
    3. ``kagglehub.model_download(<handle>)`` — REQUIRES Kaggle credentials.

**Creds / honesty gate.** Google's Perch 2 Kaggle model is consent-gated; an
anonymous download fails. So when there is no ``model_path``, no
``PERCH_V2_MODEL_PATH``, AND no Kaggle credentials (``KAGGLE_USERNAME`` +
``KAGGLE_KEY`` env, or ``~/.kaggle/kaggle.json``, or ``KAGGLE_CONFIG_DIR``), the
constructor raises ``RuntimeError`` **up-front, before any network access**. It
NEVER falls back to Perch 1 — doing so would silently corrupt the embedding
dimension (1280 vs 1536) and the model provenance.

TensorFlow + kagglehub are NOT in ``requirements-pipeline.txt`` and are imported
lazily inside ``_load`` (Lesson 11). Importing this module never pulls
TensorFlow or kagglehub. If TF is unavailable at call time, inference raises
``RuntimeError`` rather than failing silently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from faun import audio
from faun.classification import Prediction
from faun.settings import get_settings

logger = logging.getLogger(__name__)

#: Perch 2 embedding dimensionality (NOT Perch-1's 1280).
PERCH_V2_DIM = 1536

PERCH_V2_SR = 32_000
PERCH_V2_WINDOW_S = 5
PERCH_V2_WINDOW_SAMPLES = PERCH_V2_SR * PERCH_V2_WINDOW_S  # 160_000

#: Peak amplitude Perch 2 preprocessing normalizes to.
PERCH_V2_PEAK = 0.25

#: Kaggle model handles. GPU handle is the default; the *_cpu variant is the
#: cluster CPU-only build (TF without CUDA).
PERCH_V2_HANDLE_GPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
PERCH_V2_HANDLE_CPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"


def _kaggle_creds_present() -> bool:
    """True iff Kaggle credentials are discoverable without the network.

    Checks, in order: ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars, a
    ``kaggle.json`` under ``KAGGLE_CONFIG_DIR``, or ``~/.kaggle/kaggle.json``.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir and (Path(config_dir) / "kaggle.json").is_file():
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


class Perch2Adapter:
    """Perch 2 (bird vocalization) species classifier — 1536-dim embeddings.

    Args:
        model_path: Local SavedModel directory. Falls back to
            ``get_settings().perch_v2_model_path`` (the ``PERCH_V2_MODEL_PATH``
            env), then a credential-gated Kaggle download.
        labels: Optional class label list aligned with the ``label`` logits.
            When absent, predictions are named ``species_<i>``. Real labels live
            in ``<model_path>/assets/*.csv``.
        top_k: Maximum number of predictions returned by ``classify``.
        cpu: Use the CPU-only Kaggle handle when downloading (cluster default).

    Raises:
        RuntimeError: at construction, when neither a model path nor Kaggle
            credentials are available (the honesty/creds gate). Never falls back
            to Perch 1.
    """

    #: Advertised embedding dimensionality (so a 1280-vs-1536 mismatch is
    #: detectable by callers / embedders).
    DIM = PERCH_V2_DIM

    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        top_k: int = 5,
        cpu: bool = False,
    ) -> None:
        resolved = model_path or get_settings().perch_v2_model_path
        if not resolved and not _kaggle_creds_present():
            raise RuntimeError(
                "Perch 2 requires either a local SavedModel path "
                "(model_path arg / PERCH_V2_MODEL_PATH env) or Kaggle "
                "credentials (KAGGLE_USERNAME + KAGGLE_KEY, or "
                "~/.kaggle/kaggle.json / KAGGLE_CONFIG_DIR). The Google "
                "Perch 2 model is consent-gated, so an anonymous download "
                "fails. Refusing to fall back to Perch 1 (that would corrupt "
                "the 1536-dim embedding contract and model provenance)."
            )
        # ``model_path`` stays None when we will resolve via kagglehub at load.
        self.model_path = resolved
        self.labels = labels
        self.top_k = top_k
        self.cpu = cpu
        self._model = None

    def _resolve_path(self) -> str:
        """Resolve the SavedModel directory, downloading from Kaggle if needed.

        Raises:
            RuntimeError: if kagglehub is unavailable when a download is needed.
        """
        if self.model_path:
            return self.model_path
        try:
            import kagglehub
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise RuntimeError(
                "Perch 2 download needs 'kagglehub', which is not installed. "
                "Install it with `pip install kagglehub` (cluster only). It is "
                "intentionally excluded from requirements-pipeline.txt."
            ) from exc
        handle = PERCH_V2_HANDLE_CPU if self.cpu else PERCH_V2_HANDLE_GPU
        path = kagglehub.model_download(handle)
        self.model_path = path
        return path

    def _load(self):
        """Lazily load the Perch 2 SavedModel via ``tf.saved_model.load``.

        Raises:
            RuntimeError: if TensorFlow is not installed.
        """
        if self._model is not None:
            return self._model
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - exercised via mock
            raise RuntimeError(
                "Perch2Adapter requires TensorFlow (>= 2.20), which is not "
                "installed. Install it with `pip install 'tensorflow>=2.20'`. "
                "It is intentionally excluded from requirements-pipeline.txt "
                "and only available on the cluster image."
            ) from exc
        self._model = tf.saved_model.load(self._resolve_path())
        return self._model

    def _prepare(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Downmix to mono, resample to 32 kHz, fit to 5 s, then peak-normalize.

        Downmix / resample / fit-to-window are delegated to :mod:`faun.audio`
        (single preprocessing owner, ADR-0002); the Perch-2-specific peak
        normalization to :data:`PERCH_V2_PEAK` stays here.
        """
        wav = audio.downmix(waveform)
        wav = audio.resample(wav, sr, PERCH_V2_SR)
        wav = audio.fit_window(wav, PERCH_V2_WINDOW_SAMPLES)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0.0:
            wav = (wav / peak) * PERCH_V2_PEAK
        return wav.astype(np.float32, copy=False)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    def _infer(self, waveform: np.ndarray, sr: int):
        """Run the serving signature, returning ``(logits, embedding)`` numpy.

        Reuses the canonical serving call in
        ``experiments.wrappers.perch_v2._infer`` (which runs
        ``model.signatures['serving_default'](inputs=data)`` — the Perch 2 path,
        NOT Perch-1's ``infer_tf`` — and reads ``embedding`` + ``label`` from the
        dict). The wrapper returns ``(embedding, logits)``; this adapter returns
        ``(logits, embedding)``. The wrapper is imported lazily so importing this
        module stays TF-free.
        """
        from experiments.wrappers import perch_v2 as _perch_v2_wrapper

        model = self._load()
        wav = self._prepare(waveform, sr)
        data = wav[np.newaxis, :]
        emb, logits = _perch_v2_wrapper._infer(model, data)
        return self._to_numpy(logits), self._to_numpy(emb)

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Return the Perch 2 ``embedding`` for ``waveform`` (for few-shot).

        Returns a flat 1-D vector (length :data:`PERCH_V2_DIM` for the real
        model). The shape is taken verbatim from the model output — no silent
        coercion — so a 1280-vs-1536 mismatch is detectable by callers.

        Raises:
            RuntimeError: if TensorFlow is unavailable at call time.
        """
        _logits, embedding = self._infer(waveform, sr)
        return embedding[0] if embedding.ndim > 1 else embedding

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked species predictions for ``segment``.

        Raises:
            RuntimeError: if TensorFlow is unavailable at call time.
        """
        logits, _embedding = self._infer(segment, sr)
        scores = logits[0] if logits.ndim > 1 else logits
        labels = self.labels or [f"species_{i}" for i in range(len(scores))]
        order = np.argsort(scores)[::-1][: self.top_k]
        return [Prediction(labels[i], float(scores[i])) for i in order]
