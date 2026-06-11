"""Classification: species classifier protocol + adapters.

Phase-2 waves write concrete adapters (BirdNET, Perch, YAMNet embeddings+probe)
against the frozen ``SpeciesClassifier`` protocol. ``StubAdapter`` ships in the
skeleton so the pipeline and tests stay independent of any heavy ML dependency.

stdlib + typing only — no heavy imports here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Prediction", "SpeciesClassifier", "StubAdapter"]


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
