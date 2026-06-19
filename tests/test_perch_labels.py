"""TF-free tests for Perch 2 ``assets/labels.csv`` parsing (``_load_labels``).

The real Kaggle ``perch_v2_cpu/1`` ships ``assets/labels.csv`` whose first line
is a taxonomy/namespace header (``inat2024_fsd50k``) followed by 14795 scientific
names aligned with the ``label`` logits. These tests pin that contract with a
tiny in-tree fixture (no TensorFlow, no kagglehub, no network): the header must
be dropped, names must align with logit index, and a missing/garbage file must
fall back to ``species_<i>`` without crashing.
"""

from __future__ import annotations

import sys

import numpy as np

from faun.classification import Prediction
from faun.classification.perch_v2 import Perch2Adapter


# ---------------------------------------------------------------------------
# Fake SavedModel (mirrors tests/test_perch_v2.py; redefined to stay standalone)
# ---------------------------------------------------------------------------


class _FakeSignature:
    """``model.signatures['serving_default']`` stand-in returning numpy dicts.

    Logits descend by index (``[n, n-1, ..., 1]``) so ``argsort`` puts logit
    index 0 first — i.e. the top prediction is ``labels[0]`` (the first row
    AFTER the header). That makes the header-drop assertion unambiguous.
    """

    def __init__(self, n_classes: int = 3) -> None:
        self.n_classes = n_classes

    def __call__(self, inputs):
        n = np.asarray(inputs).shape[0]
        logits = np.tile(np.arange(self.n_classes, 0, -1, dtype=np.float32), (n, 1))
        return {
            "embedding": np.zeros((n, 1536), dtype=np.float32),
            "label": logits,
        }


class _FakeModel:
    def __init__(self, n_classes: int = 3) -> None:
        self.signatures = {"serving_default": _FakeSignature(n_classes)}


def _write_assets(model_dir, lines: list[str]) -> None:
    """Write ``<model_dir>/assets/labels.csv`` with the given raw lines."""
    assets = model_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _adapter(model_dir, n_classes: int = 3, **kw) -> Perch2Adapter:
    adapter = Perch2Adapter(model_path=str(model_dir), **kw)
    adapter._model = _FakeModel(n_classes)
    return adapter


# ---------------------------------------------------------------------------
# Header drop + alignment
# ---------------------------------------------------------------------------


def test_header_line_is_dropped(tmp_path):
    """The leading taxonomy header (``inat2024_fsd50k``) is NOT a class."""
    _write_assets(
        tmp_path,
        [
            "inat2024_fsd50k",
            "Abavorana luctuosa",
            "Abeillia abeillei",
            "Zvenella yunnana",
        ],
    )
    adapter = _adapter(tmp_path, n_classes=3)
    labels = adapter._load_labels()
    assert labels == ["Abavorana luctuosa", "Abeillia abeillei", "Zvenella yunnana"]
    assert "inat2024_fsd50k" not in labels


def test_classify_uses_real_scientific_names(tmp_path):
    """classify() names predictions with the loaded scientific names, in order."""
    _write_assets(
        tmp_path,
        ["inat2024_fsd50k", "Turdus merula", "Fringilla coelebs", "Parus major"],
    )
    adapter = _adapter(tmp_path, n_classes=3)
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert all(isinstance(p, Prediction) for p in preds)
    # Logit index 0 wins -> first row after the header.
    assert preds[0].species == "Turdus merula"
    assert {p.species for p in preds} == {
        "Turdus merula",
        "Fringilla coelebs",
        "Parus major",
    }


def test_comma_column_takes_first_field(tmp_path):
    """A stray extra column is tolerated: the first comma-field is the name."""
    _write_assets(
        tmp_path,
        [
            "inat2024_fsd50k",
            "Turdus merula,xx1",
            "Fringilla coelebs,xx2",
            "Parus major,xx3",
        ],
    )
    adapter = _adapter(tmp_path, n_classes=3)
    assert adapter._load_labels()[0] == "Turdus merula"


# ---------------------------------------------------------------------------
# Fallbacks (never crash)
# ---------------------------------------------------------------------------


def test_missing_assets_falls_back_to_species_index(tmp_path):
    """No assets/labels.csv -> _load_labels None -> classify names species_<i>."""
    adapter = _adapter(tmp_path, n_classes=3)  # tmp_path has no assets/
    assert adapter._load_labels() is None
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].species.startswith("species_")


def test_header_only_file_falls_back(tmp_path):
    """A file with only the header (no class rows) is rejected -> species_<i>."""
    _write_assets(tmp_path, ["inat2024_fsd50k"])
    adapter = _adapter(tmp_path, n_classes=3)
    assert adapter._load_labels() is None
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].species.startswith("species_")


def test_out_of_range_logit_index_falls_back(tmp_path):
    """More logits than label rows -> the surplus index uses species_<i>, no IndexError."""
    _write_assets(tmp_path, ["inat2024_fsd50k", "Turdus merula", "Fringilla coelebs"])
    adapter = _adapter(tmp_path, n_classes=3)  # 3 logits but only 2 labels
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    names = {p.species for p in preds}
    assert "Turdus merula" in names
    assert any(n.startswith("species_") for n in names)


# ---------------------------------------------------------------------------
# Precedence + caching + laziness
# ---------------------------------------------------------------------------


def test_explicit_labels_take_precedence_over_assets(tmp_path):
    """An explicit ``labels`` arg wins over the assets file."""
    _write_assets(tmp_path, ["inat2024_fsd50k", "FromFile1", "FromFile2", "FromFile3"])
    adapter = _adapter(tmp_path, n_classes=3, labels=["x", "y", "z"])
    assert adapter._load_labels() == ["x", "y", "z"]
    preds = adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert preds[0].species == "x"


def test_labels_loaded_once(tmp_path, monkeypatch):
    """The assets file is read at most once (the miss/result is cached)."""
    _write_assets(tmp_path, ["inat2024_fsd50k", "A", "B", "C"])
    adapter = _adapter(tmp_path, n_classes=3)
    import pathlib

    calls = {"n": 0}
    real_read = pathlib.Path.read_text

    def counting_read(self, *a, **k):
        if self.name == "labels.csv":
            calls["n"] += 1
        return real_read(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", counting_read)
    adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    adapter.classify(np.zeros(32_000, dtype=np.float32), sr=32_000)
    assert calls["n"] == 1


def test_load_labels_is_tf_free(tmp_path):
    """_load_labels is pure file I/O — it must not import TensorFlow."""
    sys.modules.pop("tensorflow", None)
    _write_assets(tmp_path, ["inat2024_fsd50k", "A", "B", "C"])
    adapter = Perch2Adapter(model_path=str(tmp_path))
    adapter._load_labels()
    assert "tensorflow" not in sys.modules
