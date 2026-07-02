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
from dataclasses import dataclass
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

#: Class-label asset shipped inside the SavedModel. ``labels.csv`` holds the
#: scientific names aligned 1:1 with the ``label`` logits; its sibling
#: ``perch_v2_ebird_classes.csv`` holds eBird codes for the same rows (we surface
#: scientific names). Verified against the real Kaggle ``perch_v2_cpu/1`` assets
#: (2026-06-19): 14796 lines = ONE taxonomy/namespace header (``inat2024_fsd50k``,
#: NOT a class) + 14795 class rows. ``_load_labels`` drops that header line.
PERCH_V2_LABELS_FILE = "labels.csv"

#: Known leading-header tokens in Perch 2 label assets (the first line names the
#: taxonomy/namespace, NOT a class). Real class names are binomial ("Genus
#: species"), so a no-space first token is also treated as a header — but the
#: load-bearing guard is the len(labels)==len(logits) cross-check in ``classify``.
_LABELS_HEADER_SENTINELS = frozenset({"inat2024_fsd50k", "ebird2021"})

#: Sibling asset aligned 1:1 with ``labels.csv``: each row carries the eBird code
#: for that class, or the literal ``no_ebird_code`` for the ~5089 non-bird
#: (FSD50K noise: Wind, Vehicle, Speech, …) rows of Perch 2's 14795-class head.
#: Used by the presence soft-gate (FR-003) to compute bird-mass from the SAME
#: logits — zero extra inference.
PERCH_V2_EBIRD_FILE = "perch_v2_ebird_classes.csv"

#: The sentinel value in ``perch_v2_ebird_classes.csv`` marking a non-bird class.
NO_EBIRD_CODE = "no_ebird_code"


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1-D logit vector (TF-free)."""
    s = np.asarray(scores, dtype=np.float64)
    s = s - np.max(s)
    e = np.exp(s)
    return e / np.sum(e)


def bird_presence_mass(scores: np.ndarray, bird_mask: np.ndarray) -> float:
    """Softmax probability mass on the bird classes — the FR-003 ``p_bird``.

    A scalar in [0, 1]: how much of the segment's softmax distribution lands on
    eBird-coded (bird) classes vs the FSD50K non-bird noise classes, from the
    SAME logits the classifier already produced (no extra inference). Free
    function so the math is unit-tested without TensorFlow.
    """
    probs = _softmax(scores)
    return float(np.sum(probs[np.asarray(bird_mask, dtype=bool)]))


def apply_presence_gate(p_species: float, p_bird: float, k: float) -> float:
    """``clamp01(p_species · (1 + p_bird · k))`` — FR-003 soft presence boost.

    A segment-level confidence rescale: when the segment is bird-dominated
    (``p_bird`` high) a species probability is boosted; when it is dominated by
    non-bird noise (``p_bird`` low) it is left near its softmax value. ``k`` is
    the gate strength (``FAUN_PRESENCE_GATE_K``); ``k == 0`` makes this the
    identity ``p_species`` — but at ``k == 0`` the adapter does not even reach
    this function (it returns the raw logit unchanged), see ``classify``.
    """
    return float(min(1.0, max(0.0, p_species * (1.0 + p_bird * k))))


@dataclass(frozen=True)
class RoutingResult:
    """Two-model routing signal derived from ONE Perch 2 inference (FR-R1).

    ``predictions`` is exactly what ``classify`` would return for the same
    scores. ``p_bird`` is the softmax mass on bird classes (``bird_presence_mass``);
    ``None`` means the eBird-class asset is missing/mismatched, so routing cannot
    decide and MUST fail open (caller keeps STATUS_PSEUDO). ``non_bird_top`` is the
    highest-scoring NON-bird class ``(label, logit)`` for triage, or ``None``.
    """

    predictions: list[Prediction]
    p_bird: float | None
    non_bird_top: tuple[str, float] | None


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
        # Whether the assets-label load has been attempted (caches a miss so we
        # don't re-stat the filesystem on every classify call).
        self._labels_loaded = False
        # Presence-gate bird mask + serve-time calibrator, each loaded once and
        # cached (including a cached miss -> None) so classify() stays cheap.
        self._bird_mask = None
        self._bird_mask_loaded = False
        self._calibrator = None
        self._calibrator_loaded = False

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

    def _load_labels(self) -> list[str] | None:
        """Lazily load scientific-name class labels from the SavedModel assets.

        Reads ``<model_path>/assets/labels.csv`` once and caches the result. The
        file's FIRST line is a taxonomy/namespace header (e.g. ``inat2024_fsd50k``)
        — NOT a class — so it is dropped; the remaining rows are scientific names
        aligned 1:1 with the ``label`` logits (empirically 14795 classes).

        An explicit ``labels`` constructor argument always wins. When the assets
        file is missing or unreadable this logs a warning and returns ``None`` so
        ``classify`` falls back to ``species_<i>`` rather than crashing. Pure file
        I/O — never imports TensorFlow.

        Resolution relies on ``self.model_path`` being set; in the kagglehub
        path that happens during ``_infer`` (``_load`` -> ``_resolve_path``), so
        ``classify`` calls this only AFTER inference has resolved the path.
        """
        if self.labels is not None:
            return self.labels
        if self._labels_loaded:
            return self.labels  # cached miss (None) — don't re-stat each call
        self._labels_loaded = True
        if not self.model_path:
            return None
        assets = Path(self.model_path) / "assets" / PERCH_V2_LABELS_FILE
        if not assets.is_file():
            logger.warning(
                "Perch 2 labels file not found at %s; predictions will be named "
                "species_<i>",
                assets,
            )
            return None
        try:
            text = assets.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "failed to read Perch 2 labels %s: %s; using species_<i>",
                assets,
                exc,
            )
            return None
        # One value per line; tolerate an accidental extra column by taking the
        # first comma-field. Drop blank lines.
        rows = [ln.split(",")[0].strip() for ln in text.splitlines() if ln.strip()]
        if not rows:
            logger.warning("Perch 2 labels file %s is empty; using species_<i>", assets)
            return None
        # Drop a leading taxonomy/namespace header ONLY when row 0 looks like a
        # header (a known sentinel, or a single token with no internal space —
        # real class names are binomial "Genus species"). Otherwise keep every
        # row. This avoids silently dropping a real class if a future asset
        # layout has no header line. The decisive correctness guard is the
        # len(labels) == len(logits) cross-check in ``classify``.
        first = rows[0]
        if first in _LABELS_HEADER_SENTINELS or " " not in first:
            rows = rows[1:]
        if not rows:
            logger.warning(
                "Perch 2 labels file %s has no class rows; using species_<i>", assets
            )
            return None
        self.labels = rows
        logger.info(
            "loaded %d Perch 2 species labels from %s", len(self.labels), assets
        )
        return self.labels

    def _load_bird_mask(self) -> np.ndarray | None:
        """Lazily load the boolean bird mask from the eBird-class asset (FR-003).

        Reads ``<model_path>/assets/perch_v2_ebird_classes.csv`` once and caches
        the result: a boolean array, True where the class has an eBird code (a
        bird) and False where the row is ``no_ebird_code`` (FSD50K noise). Mirrors
        ``_load_labels``' fail-safe — missing/unreadable/empty assets log a
        warning and return ``None`` so the gate becomes a no-op rather than
        crashing. Pure file I/O — never imports TensorFlow.

        The decisive correctness guard is the ``len(mask) == len(scores)``
        cross-check in ``classify`` (mirroring the label off-by-one guard), so a
        wrong-length asset disables the gate instead of misclassifying noise.
        """
        if self._bird_mask_loaded:
            return self._bird_mask  # cached (possibly a None miss)
        self._bird_mask_loaded = True
        if not self.model_path:
            return None
        assets = Path(self.model_path) / "assets" / PERCH_V2_EBIRD_FILE
        if not assets.is_file():
            logger.warning(
                "Perch 2 eBird-class file not found at %s; presence gate disabled",
                assets,
            )
            return None
        try:
            text = assets.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "failed to read Perch 2 eBird classes %s: %s; presence gate disabled",
                assets,
                exc,
            )
            return None
        rows = [ln.split(",")[0].strip() for ln in text.splitlines() if ln.strip()]
        # eBird codes are single tokens (no internal space), so the labels'
        # "no-space => header" heuristic is unsafe here; drop ONLY a known
        # sentinel header. The len==len(scores) cross-check in classify is the
        # real guard if the header layout ever differs.
        if rows and rows[0] in _LABELS_HEADER_SENTINELS:
            rows = rows[1:]
        if not rows:
            logger.warning(
                "Perch 2 eBird-class file %s has no rows; presence gate disabled",
                assets,
            )
            return None
        mask = np.array([r.lower() != NO_EBIRD_CODE for r in rows], dtype=bool)
        self._bird_mask = mask
        logger.info(
            "loaded Perch 2 bird mask from %s: %d bird / %d total classes",
            assets,
            int(mask.sum()),
            mask.size,
        )
        return mask

    def _load_calibrator(self):
        """Lazily load the serve-time temperature calibrator (FR-006-serve).

        From ``get_settings().perch_v2_calibrator_path`` (``PERCH_V2_CALIBRATOR_PATH``)
        — a pickled ``TemperatureCalibrator``. Unset -> ``None`` ->
        ``prob_calibrated`` stays ``None`` (output unchanged). A bad/unreadable
        pickle logs a warning and yields ``None`` rather than crashing the job.
        Cached (including a None miss).
        """
        if self._calibrator_loaded:
            return self._calibrator
        self._calibrator_loaded = True
        path = get_settings().perch_v2_calibrator_path
        if not path:
            return None
        try:
            from faun.retraining import load_probe

            self._calibrator = load_probe(path)
        except Exception as exc:  # noqa: BLE001 — calibration must never crash a job
            logger.warning(
                "failed to load Perch 2 calibrator from %s: %s; prob_calibrated "
                "will be null",
                path,
                exc,
            )
            self._calibrator = None
        return self._calibrator

    def classify(self, segment: np.ndarray, sr: int) -> list[Prediction]:
        """Return ranked species predictions for ``segment``.

        Predictions are named with the model's real scientific-name labels
        (``assets/labels.csv``); if those assets are unavailable the names fall
        back to ``species_<i>`` (never a crash).

        Two additive, OFF-by-default signals (ADR-0007), both computed from the
        SAME logits (no extra inference):

        - **Presence soft-gate (FR-003).** With ``FAUN_PRESENCE_GATE_K == 0`` (the
          default) the ``probability`` is the RAW logit, byte-for-byte unchanged
          (the literal early path — NOT the gate formula evaluated at ``k == 0``,
          which would silently swap the logit for a softmax probability). Only
          ``k > 0`` rescales it by the segment's bird presence.
        - **Serve-time calibration (FR-006-serve).** When a calibrator is
          configured, ``prob_calibrated`` carries ``softmax(logits / T)[i]`` in
          [0, 1]; otherwise ``None``. Computed from the RAW logits, independent of
          the presence gate. The raw ``probability`` is never overwritten.

        Raises:
            RuntimeError: if TensorFlow is unavailable at call time.
        """
        logits, _embedding = self._infer(segment, sr)
        scores = logits[0] if logits.ndim > 1 else logits
        return self._predictions_from_scores(scores)

    def _predictions_from_scores(self, scores: np.ndarray) -> list[Prediction]:
        """Build ranked predictions from a 1-D logit vector.

        Shared by ``classify`` and ``classify_with_routing`` — one inference, one
        code path, so both agree byte-for-byte. Contains the exact score->
        predictions logic (label load + off-by-one guard, argsort top-k,
        presence-gate ``_prob``, serve-time calibrator, Prediction build).
        """
        labels = self._load_labels()
        # Fail-safe against a silent off-by-one: the label list MUST be 1:1 with
        # the logits. If a future asset layout breaks that (e.g. an un-dropped
        # header, or a different model variant), refuse to mislabel — fall back
        # to species_<i> rather than reporting every bird as its neighbour.
        if labels is not None and len(labels) != len(scores):
            logger.warning(
                "Perch 2 label count (%d) != logit count (%d); naming predictions "
                "species_<i> to avoid an off-by-one mislabel",
                len(labels),
                len(scores),
            )
            labels = None
        order = np.argsort(scores)[::-1][: self.top_k]

        def _name(i: int) -> str:
            if labels is not None and i < len(labels):
                return labels[i]
            return f"species_{i}"

        settings = get_settings()

        # FR-003 presence soft-gate. k == 0 is a LITERAL no-op returning the raw
        # logit; only k > 0 (with a valid, length-matched bird mask) rescales the
        # probability by the segment's bird presence.
        k = settings.presence_gate_k
        bird_mask = None
        if k > 0:
            bird_mask = self._load_bird_mask()
            if bird_mask is not None and len(bird_mask) != len(scores):
                logger.warning(
                    "Perch 2 bird-mask length (%d) != logit count (%d); presence "
                    "gate disabled (fail-open)",
                    len(bird_mask),
                    len(scores),
                )
                bird_mask = None

        if k > 0 and bird_mask is not None:
            probs = _softmax(scores)
            # Use the unit-tested free function (not a re-inlined copy) so the
            # shipped p_bird math is exactly what tests cover. The extra softmax
            # inside it is negligible next to the TF inference above.
            p_bird = bird_presence_mass(scores, bird_mask)

            def _prob(i: int) -> float:
                return apply_presence_gate(float(probs[i]), p_bird, k)
        else:

            def _prob(i: int) -> float:
                return float(scores[i])  # default: raw logit, byte-for-byte unchanged

        # FR-006-serve calibration. From the RAW logits, independent of the gate.
        from faun.retraining import apply_calibration

        calibrator = self._load_calibrator()
        calibrated = (
            apply_calibration(calibrator, scores) if calibrator is not None else None
        )

        def _calib(i: int) -> float | None:
            return float(calibrated[i]) if calibrated is not None else None

        return [
            Prediction(_name(int(i)), _prob(int(i)), prob_calibrated=_calib(int(i)))
            for i in order
        ]

    def classify_with_routing(self, segment: np.ndarray, sr: int) -> RoutingResult:
        """Classify AND compute the routing ``p_bird`` from the SAME inference.

        One ``_infer`` call — never double-infer. Predictions are identical to
        ``classify`` for the same scores. If the bird mask is missing or its
        length disagrees with the logits, ``p_bird``/``non_bird_top`` are ``None``
        (fail-open: the caller must NOT reject).

        Raises:
            RuntimeError: if TensorFlow is unavailable at call time.
        """
        logits, _embedding = self._infer(segment, sr)
        scores = logits[0] if logits.ndim > 1 else logits
        preds = self._predictions_from_scores(scores)

        mask = self._load_bird_mask()
        if mask is None or len(mask) != len(scores):
            if mask is not None:
                logger.warning(
                    "Perch 2 bird-mask length (%d) != logit count (%d); routing "
                    "disabled for this segment (fail-open)",
                    len(mask),
                    len(scores),
                )
            return RoutingResult(preds, None, None)

        p_bird = bird_presence_mass(scores, mask)

        non_bird_top = None
        non_bird_idx = np.where(~mask)[0]
        if non_bird_idx.size:
            j = int(non_bird_idx[int(np.argmax(scores[non_bird_idx]))])
            labels = self._load_labels()
            if labels is not None and len(labels) != len(scores):
                labels = None
            name = (
                labels[j]
                if (labels is not None and j < len(labels))
                else f"species_{j}"
            )
            non_bird_top = (name, float(scores[j]))

        return RoutingResult(preds, p_bird, non_bird_top)
