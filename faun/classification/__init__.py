"""Classification: species classifier protocol + adapters.

Phase-2 waves write concrete adapters (BirdNET, Perch, YAMNet embeddings+probe)
against the frozen ``SpeciesClassifier`` protocol. ``StubAdapter`` ships in the
skeleton so the pipeline and tests stay independent of any heavy ML dependency.

stdlib + typing only — no heavy imports here.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "Prediction",
    "SpeciesClassifier",
    "StubAdapter",
    "BirdNETAdapter",
    "YAMNetAdapter",
    "PerchAdapter",
    "Perch2Adapter",
    "PerchProbeAdapter",
    "MaskedClassifier",
    "load_allowlist",
    "RESERVE_CHECKLIST_PATH",
]

# Lazy adapter re-exports (PEP 562). Concrete adapters live in sibling modules
# that may import heavy ML deps; we expose them at package level via __getattr__
# WITHOUT forcing those imports at ``import faun.classification`` time.
_LAZY_ADAPTERS = {
    "BirdNETAdapter": "birdnet",
    "YAMNetAdapter": "yamnet",
    "PerchAdapter": "perch",
    "Perch2Adapter": "perch_v2",
    "PerchProbeAdapter": "perch_probe",
}


def __getattr__(name: str):
    module_name = _LAZY_ADAPTERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


@dataclass
class Prediction:
    """A single species prediction for an audio segment.

    ``probability`` carries the model's own score (a raw logit for the Perch 2
    zero-shot path). ``prob_calibrated`` is an OPTIONAL, additive calibrated
    probability in [0, 1] (FR-006-serve, ADR-0007): ``None`` unless a calibrator
    is configured, sidecar-only (it never becomes a CSV column), and the raw
    ``probability`` is never overwritten. The default keeps every existing
    two-argument ``Prediction(species, probability)`` call back-compatible.
    """

    species: str
    probability: float
    prob_calibrated: float | None = None


@runtime_checkable
class SpeciesClassifier(Protocol):
    """Frozen interface every classifier adapter implements."""

    def classify(self, segment, sr) -> list[Prediction]:
        """Return ranked predictions for ``segment`` sampled at ``sr`` Hz."""
        ...


class StubAdapter:
    """Deterministic placeholder classifier (no ML deps).

    Returns fixed predictions so Phase-2 code waves can wire the pipeline
    end-to-end before real adapters exist.
    """

    def classify(self, segment, sr) -> list[Prediction]:
        return [
            Prediction("Turdus merula", 0.91),
            Prediction("unknown", 0.42),
        ]


# ---------------------------------------------------------------------------
# Regional species allow-list (MaskedClassifier) — ADR-0004
# ---------------------------------------------------------------------------

#: Fraction of allow-list names that must match the classifier's own label
#: vocabulary before the mask activates. Below this, the mask assumes a
#: name-format mismatch / wrong checklist and disables itself (fail-open) rather
#: than silently emptying the output. See ``MaskedClassifier``.
DEFAULT_COVERAGE_FLOOR = 0.5

#: ``FAUN_SPECIES_ALLOWLIST`` values that select the bundled reserve seed
#: instead of a filesystem path.
_ALLOWLIST_SENTINELS = frozenset({"default", "reserve"})

#: Bundled default reserve checklist (copied from, and kept in sync with,
#: ``scripts/extract_inatsounds_subset.py:RESERVE`` — NOT imported from scripts).
RESERVE_CHECKLIST_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "reserve_checklist.txt"
)


def _binomial(name: str) -> str:
    """Normalize a species name to ``Genus species`` form.

    Underscores become spaces and internal whitespace is collapsed, so the
    iNatSounds folder form ``Genus_species`` and the Perch / checklist form
    ``Genus species`` reconcile. Generalizes
    ``scripts/eval_inatsounds_perchv2.py:_binomial`` (which only replaced ``_``).
    Case is preserved (use :func:`_species_key` for matching).
    """
    return " ".join(str(name).replace("_", " ").split())


def _species_key(name: str) -> str:
    """Case-insensitive comparison key for a species name (binomial, casefolded)."""
    return _binomial(name).casefold()


def load_allowlist(spec: str | None) -> list[str]:
    """Load a regional species allow-list from ``spec`` (path or sentinel).

    ``spec`` is the ``FAUN_SPECIES_ALLOWLIST`` value: ``None``/blank yields an
    empty list (mask OFF); the literal ``default``/``reserve`` selects the
    bundled :data:`RESERVE_CHECKLIST_PATH`; anything else is a filesystem path.
    The file is one binomial per line; ``#`` comments and blank lines are
    skipped.

    Fail-loud, fail-open: a missing/unreadable/empty file logs a warning and
    returns ``[]`` (so the caller leaves the classifier unmasked) rather than
    raising — a checklist misconfiguration must never take the service down or
    empty the output.
    """
    if spec is None or not str(spec).strip():
        return []
    token = str(spec).strip()
    path = (
        RESERVE_CHECKLIST_PATH
        if token.casefold() in _ALLOWLIST_SENTINELS
        else Path(token)
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "species allowlist %s could not be read: %s; classifier left "
            "unmasked (no-op)",
            path,
            exc,
        )
        return []
    names = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        logger.warning(
            "species allowlist %s has no entries; classifier left unmasked (no-op)",
            path,
        )
    return names


class MaskedClassifier:
    """Restrict another classifier's predictions to a regional allow-list.

    Wraps any :class:`SpeciesClassifier` (so it works for Perch 2 zero-shot and
    a future probe head alike — applied in ``faun.api._build_classifier``, never
    inside an adapter) and drops predictions whose species is not on the
    allow-list. Names are normalized with :func:`_species_key` (case-insensitive,
    underscore-tolerant) so format differences never cause a false miss.

    Fail-loud, fail-open coverage gate: on the first ``classify`` the mask
    compares the allow-list against the wrapped classifier's own label
    vocabulary (``vocab_provider``). If fewer than ``coverage_floor`` of the
    allow-list names are present in that vocabulary — a sign of a name-format
    mismatch, a typo'd checklist, or labels that fell back to ``species_<i>`` —
    the mask logs a warning and **disables itself** (passes every prediction
    through) rather than silently emptying the CSV. Each filtered prediction is
    logged as ``masked_out=<species>``.

    Args:
        inner: the wrapped classifier (exposes ``inner`` for source-tag unwrap).
        allowlist: iterable of scientific binomials to keep.
        vocab_provider: optional zero-arg callable returning the wrapped
            classifier's full label vocabulary (e.g. ``Perch2Adapter._load_labels``).
            ``None`` -> the coverage gate cannot run -> mask stays a no-op.
        coverage_floor: minimum matched fraction to activate (default
            :data:`DEFAULT_COVERAGE_FLOOR`).
    """

    def __init__(
        self,
        inner: SpeciesClassifier,
        allowlist: Iterable[str],
        *,
        vocab_provider: Callable[[], Iterable[str] | None] | None = None,
        coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    ) -> None:
        self.inner = inner
        self.allow = frozenset(
            _species_key(s) for s in allowlist if s and str(s).strip()
        )
        self._vocab_provider = vocab_provider
        self._coverage_floor = coverage_floor
        self._checked = False
        self._active = False

    def classify(self, segment, sr) -> list[Prediction]:
        """Classify via the inner model, then keep only allow-listed species."""
        preds = self.inner.classify(segment, sr)
        if not self._ensure_active():
            return preds
        kept: list[Prediction] = []
        for pred in preds:
            if _species_key(pred.species) in self.allow:
                kept.append(pred)
            else:
                logger.info("masked_out=%s", pred.species)
        return kept

    def _ensure_active(self) -> bool:
        """Run the one-time coverage gate; return whether the mask is active."""
        if self._checked:
            return self._active
        self._checked = True
        if not self.allow:
            logger.warning(
                "species allowlist is empty; classifier left unmasked (no-op)"
            )
            return False
        vocab = self._model_vocab()
        if not vocab:
            logger.warning(
                "classifier %s exposes no label vocabulary to validate the species "
                "allowlist against; classifier left unmasked (no-op, fail-open)",
                self.inner.__class__.__name__,
            )
            return False
        vocab_keys = {_species_key(v) for v in vocab}
        matched = self.allow & vocab_keys
        coverage = len(matched) / len(self.allow)
        if coverage < self._coverage_floor:
            logger.warning(
                "species allowlist coverage %.0f%% (%d/%d names match the classifier "
                "vocabulary) is below the %.0f%% floor — likely a name-format mismatch "
                "or wrong checklist; classifier left unmasked (no-op, fail-open) rather "
                "than emptying the output",
                coverage * 100,
                len(matched),
                len(self.allow),
                self._coverage_floor * 100,
            )
            return False
        self._active = True
        logger.info(
            "species allowlist active: %d names, %.0f%% vocabulary coverage",
            len(self.allow),
            coverage * 100,
        )
        return True

    def _model_vocab(self) -> list[str] | None:
        """Best-effort fetch of the wrapped classifier's label vocabulary."""
        if self._vocab_provider is None:
            return None
        try:
            vocab = self._vocab_provider()
        except Exception:  # noqa: BLE001 — vocab probing must never break classify
            logger.warning(
                "species allowlist vocab provider raised; classifier left unmasked "
                "(no-op, fail-open)",
                exc_info=True,
            )
            return None
        return list(vocab) if vocab else None
