"""Тесты species_eval — РЕАЛЬНО прогоняют ML-путь, но БЕЗ TensorFlow.

Гейт честности: фичи синтетические (np.random, кластеризованные по классам),
проба — настоящая sklearn LogisticRegression. Каждое число, полученное на
синтетике, помечено provenance="SYNTHETIC — not a species metric".

Никакого TF/iNatSounds: real species metric существует только после прогона
на кластере (scripts/train_inatsounds.sh, synthetic=False).
"""

from __future__ import annotations

import numpy as np

from faun import retraining
from experiments.wrappers.yamnet_probe import train_probe

SYNTHETIC_TAG = "SYNTHETIC — not a species metric"


def _clustered_dataset(n_classes: int = 3, per_class: int = 40, dim: int = 16, seed=0):
    """Синтетика: каждый класс — гауссово облако вокруг своего центра.

    Кластеры разнесены, чтобы линейная проба реально что-то выучила (recall>0).
    """
    rng = np.random.default_rng(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[str] = []
    for c in range(n_classes):
        center = np.zeros(dim, dtype=float)
        center[c % dim] = 6.0  # разнесённые центры -> разделимость
        X_parts.append(rng.standard_normal((per_class, dim)) + center)
        y_parts.extend([f"species_{c}"] * per_class)
    return np.vstack(X_parts), np.asarray(y_parts)


def test_species_eval_runs_real_probe_and_tags_synthetic():
    """Гейт 1: реально фитим пробу, гоняем eval, проверяем форму и provenance."""
    X, y = _clustered_dataset(n_classes=3, per_class=40, dim=16, seed=1)
    clf = train_probe(X, y, seed=42)

    report = retraining.species_eval(clf, X, y, synthetic=True)

    # per-species recall: ключи = все классы.
    assert set(report["per_species_recall"].keys()) == set(np.unique(y))
    assert all(0.0 <= v <= 1.0 for v in report["per_species_recall"].values())

    # macro-F1 в [0,1].
    assert 0.0 <= report["macro_f1"] <= 1.0

    # confusion: квадратная матрица n_classes x n_classes.
    confusion = np.asarray(report["confusion"])
    assert confusion.shape == (3, 3)

    # labels — упорядоченный список классов, согласован с confusion.
    assert list(report["labels"]) == sorted(np.unique(y))

    # n / n_classes.
    assert report["n"] == len(y)
    assert report["n_classes"] == 3

    # ГЕЙТ честности: число помечено как синтетическое.
    assert report["provenance"] == SYNTHETIC_TAG


def test_species_eval_learns_separable_clusters():
    """Кластеры разделимы -> recall должен быть высоким (eval не пустышка)."""
    X, y = _clustered_dataset(n_classes=3, per_class=50, dim=16, seed=2)
    clf = train_probe(X, y, seed=42)

    report = retraining.species_eval(clf, X, y, synthetic=True)

    # train-recall на разделимых кластерах ожидаемо высокий.
    assert report["macro_f1"] > 0.8
    assert min(report["per_species_recall"].values()) > 0.6


def test_species_eval_includes_cv_value_and_ci():
    """Отчёт несёт CV-оценку с CI (переиспользует train_probe_cv)."""
    X, y = _clustered_dataset(n_classes=3, per_class=120, dim=16, seed=3)
    clf = train_probe(X, y, seed=42)

    report = retraining.species_eval(clf, X, y, synthetic=True)

    # value присутствует; при достаточном n — CI заполнен.
    assert "value" in report
    assert report["ci_low"] is not None and report["ci_high"] is not None
    assert report["ci_low"] <= report["value"] <= report["ci_high"]


def test_species_eval_non_synthetic_tag():
    """synthetic=False -> provenance НЕ помечен как синтетический."""
    X, y = _clustered_dataset(n_classes=2, per_class=30, dim=8, seed=4)
    clf = train_probe(X, y, seed=42)

    report = retraining.species_eval(clf, X, y, synthetic=False)

    assert report["provenance"] != SYNTHETIC_TAG
    assert "SYNTHETIC" not in report["provenance"]


def test_species_eval_confusion_diagonal_dominant_when_separable():
    """Диагональ confusion доминирует на разделимых данных."""
    X, y = _clustered_dataset(n_classes=4, per_class=40, dim=16, seed=5)
    clf = train_probe(X, y, seed=42)

    report = retraining.species_eval(clf, X, y, synthetic=True)
    confusion = np.asarray(report["confusion"])

    # На каждой строке диагональ — максимум (класс предсказан верно чаще всего).
    for i in range(confusion.shape[0]):
        assert confusion[i, i] == confusion[i].max()


def test_species_eval_rejects_dim_mismatch():
    """Гейт: проба и эмбеддинги разной размерности -> явный ValueError, не молча.

    Регрессия на YAMNet-несовместимость (YAMNetAdapter.embed=1024 vs
    YamnetEmbedder=2048): скрестив их, sklearn упал бы невнятно — мы падаем явно.
    """
    import pytest

    X, y = _clustered_dataset(n_classes=2, per_class=20, dim=4, seed=3)
    clf = train_probe(X, y, seed=42)  # n_features_in_ == 4
    X_wrong = np.zeros((len(y), 8), dtype=float)  # 8 != 4
    with pytest.raises(ValueError, match="same embedder|expects 4-dim"):
        retraining.species_eval(clf, X_wrong, y, synthetic=True)
