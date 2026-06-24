"""Human-label -> retrain -> save -> deploy loop for the YAMNet species probe.

Ground truth for training is **only** human labels (expert ornithologist /
operator ranger) whose status is ``confirmed`` or ``corrected``. Model
pseudo-labels (``model:*`` / status ``pseudo``) are discarded — a hard negative
gate enforced by :func:`filter_ground_truth` and re-asserted in
:func:`retrain_from_labels` (explicit refusal, never an empty fit).

The probe is a scikit-learn ``LogisticRegression`` (reusing
``experiments.wrappers.yamnet_probe.train_probe``) over frozen YAMNet
embeddings. TensorFlow is pulled lazily, only when embedding real audio on the
cluster; nothing here imports TF at module level, and the label gate runs
before any ``model.embed`` call so the negative gate is testable TF-free.

The saved pickle mirrors ``faun.classification.yamnet.YAMNetAdapter._load_probe``
so a retrained probe is loadable via the ``YAMNET_PROBE_PATH`` env var.
"""

from __future__ import annotations

import math
import pickle  # noqa: S403  -- operator-supplied local model file, not network input
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Single home for the ground-truth predicate — imported, not mirrored.
# (faun.detections does NOT import faun.retraining, so this is acyclic.)
from faun.detections import is_ground_truth


def _field(label: Any, key: str) -> Any:
    """Read ``key`` from a label given as a mapping or an attribute object."""
    if isinstance(label, dict):
        return label.get(key)
    return getattr(label, key, None)


def filter_ground_truth(labels) -> list:
    """Keep only ground-truth labels; drop ``model:*`` / pseudo entries."""
    return [label for label in labels if is_ground_truth(label)]


class _ConstantProbe:
    """Degenerate single-class probe (LogisticRegression needs >= 2 classes).

    Picklable and exposing the sklearn-ish ``classes_`` / ``predict_proba`` /
    ``predict`` surface so it stays loadable via ``YAMNetAdapter`` without TF.
    """

    def __init__(self, label) -> None:
        self.classes_ = np.array([] if label is None else [label])

    def predict_proba(self, X):
        return np.ones((len(np.asarray(X)), 1), dtype=float)

    def predict(self, X):
        return np.repeat(self.classes_, len(np.asarray(X)))


def train_probe_cv(X, y, *, seed: int = 42, min_cv_n: int = 300):
    """Fit a probe and report a cross-validated metric with a 95% CI.

    Fits ``clf`` via ``train_probe(X, y, seed)``. When ``len(y) >= min_cv_n`` and
    every class has at least 2 samples, runs StratifiedKFold cross-validation
    (``roc_auc`` for 2 classes, else ``accuracy``) and a 95% CI
    (``mean +- 1.96 * std / sqrt(k)``). For small samples it falls back to a
    point estimate (CV mean if feasible, else train score) with null CI bounds
    and a note flagging the unreliable CI. Folds are clamped to
    ``min(5, smallest_class_count)``; CV is skipped when the smallest class has
    fewer than 2 samples. Never crashes on tiny ``n``.

    Returns ``(clf, metrics)`` where ``metrics`` has keys ``n``, ``n_classes``,
    ``metric``, ``value``, ``ci_low``, ``ci_high``, ``note``.
    """
    from experiments.wrappers.yamnet_probe import train_probe

    X = np.asarray(X)
    y = np.asarray(y)
    n = int(len(y))
    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(len(classes))
    smallest_class = int(counts.min()) if n_classes else 0
    metric = "roc_auc" if n_classes == 2 else "accuracy"

    note = ""
    ci_low: float | None = None
    ci_high: float | None = None

    if n_classes < 2:
        # A single-class probe cannot be fit by LogisticRegression; refuse to
        # crash and report a degenerate train-accuracy point estimate.
        clf = _ConstantProbe(classes[0] if n_classes else None)
        value = 1.0
        metric = "train_accuracy"
        note = f"n small — CI unreliable (n={n})"
        return clf, {
            "n": n,
            "n_classes": n_classes,
            "metric": metric,
            "value": float(value),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "note": note,
        }

    clf = train_probe(X, y, seed)
    cv_feasible = smallest_class >= 2

    if n >= min_cv_n and cv_feasible:
        value, ci_low, ci_high = _cv_score(X, y, seed, metric, smallest_class)
    elif cv_feasible:
        # Small sample: CV point estimate but CI is not trustworthy.
        value, _, _ = _cv_score(X, y, seed, metric, smallest_class)
        note = f"n small — CI unreliable (n={n})"
    else:
        # Cannot cross-validate (a class has <2 samples): use the train score.
        value = float(clf.score(X, y))
        metric = "train_accuracy"
        note = f"n small — CI unreliable (n={n})"

    return clf, {
        "n": n,
        "n_classes": n_classes,
        "metric": metric,
        "value": float(value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "note": note,
    }


def _cv_score(X, y, seed: int, metric: str, smallest_class: int):
    """StratifiedKFold CV -> (mean, ci_low, ci_high) for the chosen metric."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from experiments.wrappers.yamnet_probe import train_probe

    folds = min(5, smallest_class)
    folds = max(2, folds)
    estimator = train_probe(X, y, seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=metric)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    half = 1.96 * std / math.sqrt(folds)
    return mean, mean - half, mean + half


# ---------------------------------------------------------------------------
# Prototypical probe with a negative (background) class (FR-007, ADR-0008)
# ---------------------------------------------------------------------------
#
# Why this exists: a plain ``LogisticRegression`` / softmax probe over Perch 2
# embeddings is a CLOSED-WORLD classifier — every input is forced onto one of the
# trained bird species, so an out-of-distribution embedding (wind, vehicle,
# silence, an untrained species) maps to a *confident* bird. The served
# ``PerchProbeAdapter`` would then surface that confident-but-wrong bird.
#
# A prototypical probe stores one centroid per class and scores a query by its
# distance to each centroid (softmax over negative squared distances, or cosine
# similarity, scaled by a temperature). Adding a distinct NEGATIVE class with its
# own centroid built from non-bird (background) embeddings gives the probe an
# explicit "none of the birds" bucket: an OOD embedding near the negative
# prototype is classified :data:`NEGATIVE_CLASS` instead of a bird.
#
# It is a drop-in for the EXISTING ``PerchProbeAdapter`` — it exposes the same
# ``predict_proba(X) -> (N, C)`` / ``classes_`` surface as the sklearn probes and
# pickles cleanly (pure numpy state, no closures), so the adapter and
# ``save_probe`` / ``load_probe`` work UNCHANGED. Deterministic, torch-free.

#: Label of the explicit background / out-of-distribution bucket. Distinct from
#: the ``"unknown"`` species token (``StubAdapter``) and from ``STATUS_REJECTED``
#: (a label lifecycle status) — this is a real *class* the probe predicts.
NEGATIVE_CLASS = "__negative__"


class _PrototypeProbe:
    """Nearest-centroid probe with optional negative class. Picklable, TF-free.

    Stores one prototype (centroid) per class. ``predict_proba`` returns a
    softmax over per-class similarity logits, so rows sum to 1 and lie in
    ``[0, 1]`` — the contract ``PerchProbeAdapter`` / ``YAMNetAdapter`` rely on.
    State is pure numpy (``classes_`` / ``prototypes_`` / scalars), so it pickles
    without any closure or third-party object.

    Args:
        metric: ``"cosine"`` (L2-normalize embeddings + prototypes, similarity =
            dot product) or ``"euclidean"`` (logit = negative squared distance).
        temperature: positive scalar dividing the similarity logits before
            softmax (sharper for small ``T``). Non-finite / non-positive values
            fall back to ``1.0``.
    """

    def __init__(self, metric: str = "cosine", temperature: float = 1.0) -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")
        self.metric = metric
        t = float(temperature)
        self.temperature = t if (math.isfinite(t) and t > 0) else 1.0
        self.classes_: np.ndarray = np.array([])
        self.prototypes_: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.n_features_in_: int | None = None

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)  # leave all-zero rows untouched
        return mat / norms

    def fit(self, X, y, *, negatives=None) -> "_PrototypeProbe":
        """Build per-class centroids from ``(X, y)`` (+ optional ``negatives``).

        ``negatives`` (``(M, D)`` non-bird embeddings) get their own centroid
        under the :data:`NEGATIVE_CLASS` label. Classes are sorted so the column
        order is deterministic and matches ``classes_``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D); got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")
        if len(X) == 0:
            raise ValueError("cannot fit a prototype probe on an empty X")

        self.n_features_in_ = int(X.shape[1])

        labels = sorted({str(v) for v in y.tolist()})
        centroids = [
            X[np.asarray([str(v) for v in y.tolist()]) == lbl].mean(axis=0)
            for lbl in labels
        ]

        if negatives is not None:
            neg = np.asarray(negatives, dtype=np.float64)
            if neg.ndim != 2 or neg.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"negatives must be (M, {self.n_features_in_}) to match X; "
                    f"got shape {neg.shape}"
                )
            if len(neg) == 0:
                raise ValueError("negatives given but empty; pass None for no negative")
            labels.append(NEGATIVE_CLASS)
            centroids.append(neg.mean(axis=0))

        self.classes_ = np.array(labels)
        prototypes = np.stack(centroids).astype(np.float64)
        if self.metric == "cosine":
            prototypes = self._l2_normalize(prototypes)
        self.prototypes_ = prototypes
        return self

    def _logits(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if self.prototypes_.shape[0] == 0:
            raise RuntimeError("probe is not fitted (no prototypes)")
        if self.metric == "cosine":
            sims = self._l2_normalize(X) @ self.prototypes_.T  # cosine in [-1, 1]
        else:
            # -||x - c||^2 = 2 x·c - ||x||^2 - ||c||^2 ; the per-row ||x||^2 is a
            # constant shift that softmax cancels, so only 2 x·c - ||c||^2 matters.
            xc = X @ self.prototypes_.T
            cc = np.sum(self.prototypes_**2, axis=1)
            sims = 2.0 * xc - cc[np.newaxis, :]
        return sims / self.temperature

    def predict_proba(self, X) -> np.ndarray:
        """Per-class probabilities ``(N, C)`` (softmax over similarity logits)."""
        return _softmax(self._logits(X), axis=-1)

    def predict(self, X) -> np.ndarray:
        """Argmax class label per row (aligned with ``classes_``)."""
        idx = np.argmax(self._logits(X), axis=1)
        return self.classes_[idx]


def train_prototype_probe(
    X,
    y,
    *,
    negatives=None,
    metric: str = "cosine",
    temperature: float = 1.0,
):
    """Fit a :class:`_PrototypeProbe` (per-class centroids) over embeddings.

    A deterministic, torch-free alternative to ``train_probe_cv``'s
    ``LogisticRegression``: it stores one centroid per class and scores by
    similarity, so it admits an explicit NEGATIVE class. The returned probe is a
    drop-in for the UNCHANGED ``PerchProbeAdapter`` (same ``predict_proba`` /
    ``classes_`` surface, picklable via :func:`save_probe`).

    Args:
        X: ``(N, D)`` embeddings (dimension-agnostic — works for Perch 2's 1536
            as well as small synthetic dims).
        y: length-``N`` species labels.
        negatives: optional ``(M, D)`` non-bird (background / OOD) embeddings.
            When given, a distinct :data:`NEGATIVE_CLASS` prototype is added so an
            OOD embedding near it is classified negative rather than as a
            confident bird. ``None`` -> a plain multiclass prototypical probe.
        metric: ``"cosine"`` (default) or ``"euclidean"`` (see :class:`_PrototypeProbe`).
        temperature: softmax temperature over the similarity logits.

    Returns:
        A fitted :class:`_PrototypeProbe`.
    """
    probe = _PrototypeProbe(metric=metric, temperature=temperature)
    probe.fit(X, y, negatives=negatives)
    return probe


def save_probe(clf, out_path) -> Path:
    """Pickle ``clf`` to ``out_path`` (sklearn probe; no TF).

    Mirrors ``YAMNetAdapter._load_probe`` so the file is loadable by
    ``YAMNetAdapter(probe_path=...)`` / ``YAMNET_PROBE_PATH``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(clf, fh)
    return out_path


def load_probe(path):
    """Unpickle a probe saved by :func:`save_probe` (no TF on this path)."""
    with open(Path(path), "rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _resolve_clip(label: Any, audio_dir: Path) -> Path | None:
    """Resolve a label's clip path under ``audio_dir``.

    Prefers an explicit ``segment_path`` (relative to ``audio_dir``); otherwise
    falls back to ``source_file`` in the same directory. Returns ``None`` when
    neither resolves to an existing file.
    """
    seg = _field(label, "segment_path")
    if seg:
        candidate = audio_dir / str(seg)
        if candidate.exists():
            return candidate
    src = _field(label, "source_file")
    if src:
        candidate = audio_dir / str(src)
        if candidate.exists():
            return candidate
    return None


def _read_clip(path: Path):
    """Load a clip as ``(waveform, sr)`` via soundfile."""
    import soundfile as sf

    waveform, sr = sf.read(str(path))
    return np.asarray(waveform, dtype=np.float32), int(sr)


def retrain_from_labels(labels, audio_dir, model, out_path) -> dict:
    """Retrain the probe from human ground-truth labels and save it.

    Filters to ground-truth (expert/operator) labels; if none remain, raises
    ``ValueError`` *before* touching ``model`` (so the negative gate is testable
    without TensorFlow). Otherwise embeds each clip via ``model.embed`` (TF pulled
    lazily on the cluster), trains a cross-validated probe, saves it, and returns
    the metrics augmented with ``out_path``.
    """
    filtered = filter_ground_truth(labels)
    if not filtered:
        raise ValueError("no ground-truth (expert/ranger) labels to train on")

    audio_dir = Path(audio_dir)
    rows: list[np.ndarray] = []
    species: list[str] = []
    skipped = 0
    for label in filtered:
        clip = _resolve_clip(label, audio_dir)
        if clip is None:
            skipped += 1
            continue
        waveform, sr = _read_clip(clip)
        embedding = np.asarray(model.embed(waveform, sr))
        rows.append(embedding)
        species.append(str(_field(label, "species")))

    if not rows:
        raise ValueError("no resolvable ground-truth clips found under audio_dir")

    X = np.stack(rows)
    y = np.asarray(species)
    clf, metrics = train_probe_cv(X, y)
    save_probe(clf, out_path)
    metrics["out_path"] = str(out_path)
    if skipped:
        suffix = f"{skipped} label(s) skipped (clip not found)"
        metrics["note"] = f"{metrics['note']}; {suffix}" if metrics["note"] else suffix
    return metrics


# Метка происхождения числа, полученного на СИНТЕТИЧЕСКИХ эмбеддингах: это НЕ
# species-метрика. Реальное число существует только после прогона на кластере
# на настоящем iNatSounds (scripts/train_inatsounds.sh, synthetic=False).
_SYNTHETIC_PROVENANCE = "SYNTHETIC — not a species metric"


def species_eval(clf, X, y, *, synthetic: bool = True) -> dict:
    """Оценить мультиклассовую пробу по видам: recall, macro-F1, confusion + CV.

    Считает per-species recall (``recall_score(average=None)``), macro-F1
    (``f1_score(average='macro')``) и матрицу ошибок (``confusion_matrix``) на
    переданных ``(X, y)``, а также CV-оценку с 95% CI, переиспользуя
    :func:`train_probe_cv` (логика кросс-валидации не дублируется).

    Гейт честности: при ``synthetic=True`` (по умолчанию — путь юнит-тестов и
    синтетических эмбеддингов) ключ ``provenance`` равен
    ``"SYNTHETIC — not a species metric"``. Реальное species-число получают
    только с ``synthetic=False`` на настоящем iNatSounds (на кластере).

    Чистый numpy/sklearn — TensorFlow здесь не импортируется.

    Returns ``dict`` с ключами как минимум: ``per_species_recall`` (dict),
    ``macro_f1`` (float), ``confusion`` (2D list), ``labels`` (упорядоченный
    список классов), ``n``, ``n_classes``, ``provenance``, ``note`` (+ CV-поля
    ``metric``/``value``/``ci_low``/``ci_high`` из :func:`train_probe_cv`).
    """
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    X = np.asarray(X)
    y = np.asarray(y)
    labels = sorted(np.unique(y).tolist())

    # Гейт совместимости размерностей: проба и эмбеддер ОБЯЗАНЫ совпадать по DIM.
    # YAMNet даёт два несовместимых вектора — YAMNetAdapter.embed=1024 (mean) и
    # YamnetEmbedder=2048 (concat(mean,max)); скрестив их, sklearn упал бы с
    # невнятной ошибкой. Падаем явно. См. docs/training.md.
    n_features = getattr(clf, "n_features_in_", None)
    if n_features is not None and X.ndim == 2 and X.shape[1] != n_features:
        raise ValueError(
            f"probe expects {n_features}-dim features but X has {X.shape[1]} — "
            "train and eval must use the SAME embedder "
            "(YamnetEmbedder=2048 vs YAMNetAdapter.embed=1024). See docs/training.md."
        )

    # Гейт словаря (как dim-guard, но по классам): если у пробы есть classes_ и
    # НИ ОДИН вид eval-набора в него не входит — это почти наверняка чужая проба/
    # датасет, recall был бы тождественным нулём и «честным» молчанием. Падаем
    # явно ТОЛЬКО при ПОЛНОМ непересечении; частичное пересечение легитимно
    # (новые виды на eval допустимы).
    clf_classes = getattr(clf, "classes_", None)
    if clf_classes is not None and len(clf_classes) and len(y):
        eval_species = set(map(str, np.asarray(y).tolist()))
        probe_species = set(map(str, np.asarray(clf_classes).tolist()))
        if eval_species.isdisjoint(probe_species):
            raise ValueError(
                "probe vocabulary is disjoint from the eval labels "
                f"(probe {sorted(probe_species)[:3]}… vs eval "
                f"{sorted(eval_species)[:3]}…) — train and eval must use the SAME "
                "dataset vocabulary."
            )

    y_pred = clf.predict(X)

    recalls = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)
    per_species_recall = {
        species: float(value) for species, value in zip(labels, recalls)
    }
    macro_f1 = float(
        f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)
    )
    confusion = confusion_matrix(y, y_pred, labels=labels).tolist()

    # CV-оценка с CI — переиспользуем общий контур, не дублируя StratifiedKFold.
    _clf_cv, cv_metrics = train_probe_cv(X, y)

    provenance = _SYNTHETIC_PROVENANCE if synthetic else "real-eval"

    return {
        "per_species_recall": per_species_recall,
        "macro_f1": macro_f1,
        "confusion": confusion,
        "labels": labels,
        "n": int(len(y)),
        "n_classes": len(labels),
        "provenance": provenance,
        "metric": cv_metrics["metric"],
        "value": cv_metrics["value"],
        "ci_low": cv_metrics["ci_low"],
        "ci_high": cv_metrics["ci_high"],
        "note": cv_metrics["note"],
    }


# ---------------------------------------------------------------------------
# Probability calibration (FR-006, ADR-0005)
# ---------------------------------------------------------------------------
#
# Why this exists: the served zero-shot ``Perch2Adapter`` reports RAW LOGITS in
# the ``probability`` field (values > 1 — observed on raw180). A logit is not a
# probability. Temperature scaling (Guo et al. 2017) maps a logit vector to a
# calibrated distribution via ``softmax(logits / T)`` with a single scalar ``T``
# fit by minimising NLL on a held-out labelled set; ``T = 1`` is plain softmax.
#
# Honesty contract: the raw ``probability`` is NEVER overwritten and the CSV
# columns stay frozen — a calibrated value travels as the OPTIONAL sidecar field
# ``Label.prob_calibrated`` (``detections.jsonl`` only). The identity default
# (:func:`apply_calibration` with ``calibrator=None``) is a pure pass-through, so
# nothing invents a calibrated number until a calibrator is explicitly fit and
# configured.


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class TemperatureCalibrator:
    """Single-scalar temperature scaling: ``p = softmax(logits / T)``.

    ``temperature == 1.0`` is plain softmax (the un-fit / neutral state).
    ``classes`` records the class order the logit columns correspond to, so a
    caller can align a per-class logit vector; it is informational only — the
    transform itself is class-agnostic.
    """

    temperature: float = 1.0
    classes: list | None = None

    def apply(self, logits) -> np.ndarray:
        """Map a logit vector / matrix to calibrated probabilities (softmax/T)."""
        t = float(self.temperature)
        if not math.isfinite(t) or t <= 0:
            t = 1.0
        return _softmax(np.asarray(logits, dtype=np.float64) / t, axis=-1)


def fit_temperature(
    logits,
    y,
    *,
    classes=None,
    bounds: tuple[float, float] = (0.05, 100.0),
) -> TemperatureCalibrator:
    """Fit a :class:`TemperatureCalibrator` by minimising multiclass NLL.

    Args:
        logits: ``(N, C)`` raw scores (one row per sample, one column per class).
        y: length-``N`` true labels. If ``classes`` is given, ``y`` values are
            matched against it to find the gold column; otherwise ``y`` is taken
            as 0-based integer column indices.
        classes: optional ordered class labels for the ``C`` columns.
        bounds: search interval for ``T`` (must be positive).

    Returns:
        A fitted :class:`TemperatureCalibrator`. With a single class, or if the
        optimiser fails, falls back to ``T = 1.0`` (plain softmax) rather than
        raising — calibration must never crash a job.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C); got shape {logits.shape}")
    n, c = logits.shape

    if classes is not None:
        classes = list(classes)
        index = {cls: i for i, cls in enumerate(classes)}
        try:
            gold = np.array([index[v] for v in np.asarray(y).tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(
                f"label {exc} in y is absent from classes — y and classes must "
                "share the same vocabulary"
            ) from None
    else:
        gold = np.asarray(y, dtype=int)
        classes = list(range(c))

    if c < 2 or n == 0:
        return TemperatureCalibrator(temperature=1.0, classes=classes)

    rows = np.arange(n)

    def _nll(t: float) -> float:
        t = max(float(t), 1e-6)
        logp = np.log(np.clip(_softmax(logits / t, axis=-1), 1e-12, 1.0))
        return float(-np.mean(logp[rows, gold]))

    result = minimize_scalar(_nll, bounds=bounds, method="bounded")
    t = float(result.x) if getattr(result, "success", True) else 1.0
    if not math.isfinite(t) or t <= 0:
        t = 1.0
    return TemperatureCalibrator(temperature=t, classes=classes)


def apply_calibration(calibrator, logits) -> np.ndarray:
    """Apply ``calibrator`` to ``logits``; identity (raw pass-through) if ``None``.

    The identity default is the honesty guard: with no calibrator configured the
    scores are returned unchanged (as a float array), so the pipeline never
    fabricates a calibrated probability.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if calibrator is None:
        return arr
    return calibrator.apply(arr)


def expected_calibration_error(
    probs,
    y_true,
    *,
    classes=None,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error of top-1 confidence (lower is better).

    Bins predictions by their max-probability (confidence) into ``n_bins`` equal
    buckets and returns the sample-weighted mean gap between accuracy and mean
    confidence per bucket. ``classes`` maps ``probs`` columns to label values
    when ``y_true`` carries labels rather than column indices.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    pred_idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(probs.shape[0]), pred_idx]

    if classes is not None:
        classes = list(classes)
        index = {cls: i for i, cls in enumerate(classes)}
        gold = np.array(
            [index.get(v, -1) for v in np.asarray(y_true).tolist()], dtype=int
        )
    else:
        gold = np.asarray(y_true, dtype=int)
    correct = (pred_idx == gold).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        # last bin is closed on the right so conf == 1.0 is counted
        in_bin = (
            (conf > lo) & (conf <= hi)
            if hi < 1.0
            else (conf > lo) & (conf <= hi + 1e-9)
        )
        m = int(np.sum(in_bin))
        if m == 0:
            continue
        ece += (m / n) * abs(
            float(np.mean(correct[in_bin])) - float(np.mean(conf[in_bin]))
        )
    return float(ece)
