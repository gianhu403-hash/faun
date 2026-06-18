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
from pathlib import Path
from typing import Any

import numpy as np

# Shared ground-truth semantics (mirrored, NOT imported from faun.detections so
# this module stays decoupled from the parallel Phase-2 build).
_GROUND_TRUTH_SOURCE_PREFIXES = ("expert:", "operator:")
_GROUND_TRUTH_STATUSES = frozenset({"confirmed", "corrected"})


def _field(label: Any, key: str) -> Any:
    """Read ``key`` from a label given as a mapping or an attribute object."""
    if isinstance(label, dict):
        return label.get(key)
    return getattr(label, key, None)


def is_ground_truth(label: Any) -> bool:
    """True iff ``label`` is a human ground-truth label.

    Ground truth IFF ``source`` starts with ``expert:`` or ``operator:`` AND
    ``status`` is ``confirmed`` or ``corrected``. Accepts dicts (``["source"]``
    / ``["status"]``) or objects (``.source`` / ``.status``). Model sources
    (``model:*``, always ``pseudo``) are never ground truth.
    """
    source = _field(label, "source")
    status = _field(label, "status")
    if not isinstance(source, str) or not isinstance(status, str):
        return False
    return (
        source.startswith(_GROUND_TRUTH_SOURCE_PREFIXES)
        and status in _GROUND_TRUTH_STATUSES
    )


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
