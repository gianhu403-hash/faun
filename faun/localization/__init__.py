"""Localization: TDOA multilateration of acoustic events across traps.

Ported from the v1 hackathon edge core. Method is hyperbolic multilateration
(growing circles ``c * (tau - t0)`` whose intersection is the source), validated
on SYNTHETIC signals only — real trap timestamps are second-grained, roughly
1500x too coarse to resolve the sub-millisecond inter-trap delays, so real-data
localization is blocked pending clock sync. A localized result therefore carries
the honest ``"tdoa-synthetic-validated"`` method tag.

Heavy deps are scipy/numpy only — no TensorFlow, no ML imports here.
"""

from __future__ import annotations

from faun.localization.distance import estimate_distances
from faun.localization.triangulate import (
    MicPosition,
    TriangulationResult,
    localize_event,
    triangulate,
)

__all__ = [
    "MicPosition",
    "TriangulationResult",
    "triangulate",
    "estimate_distances",
    "localize_event",
]
