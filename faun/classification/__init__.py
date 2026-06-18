"""Classification: species classifier protocol + adapters.

Phase-2 waves write concrete adapters (BirdNET, Perch, YAMNet embeddings+probe)
against the frozen ``SpeciesClassifier`` protocol. ``StubAdapter`` ships in the
skeleton so the pipeline and tests stay independent of any heavy ML dependency.

stdlib + typing only — no heavy imports here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "Prediction",
    "SpeciesClassifier",
    "StubAdapter",
    "BirdNETAdapter",
    "YAMNetAdapter",
    "PerchAdapter",
    "Perch2Adapter",
]

# Lazy adapter re-exports (PEP 562). Concrete adapters live in sibling modules
# that may import heavy ML deps; we expose them at package level via __getattr__
# WITHOUT forcing those imports at ``import faun.classification`` time.
_LAZY_ADAPTERS = {
    "BirdNETAdapter": "birdnet",
    "YAMNetAdapter": "yamnet",
    "PerchAdapter": "perch",
    "Perch2Adapter": "perch_v2",
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
    """A single species prediction for an audio segment."""

    species: str
    probability: float


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
