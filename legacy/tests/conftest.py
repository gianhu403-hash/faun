"""Shared fixtures for legacy (v1-hackathon) tests.

NOTE: legacy tests are frozen at tag v1-hackathon and are NOT run by CI
(CI runs only tests/). This conftest carries the TDOA / triangulation and
audio fixtures the moved hackathon tests rely on. Imports here may reference
legacy.edge.* and are not guaranteed green.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from faun.ml.yamnet import AudioResult, AudioClass
from legacy.edge.tdoa.triangulate import MicPosition, TriangulationResult


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate used across the project."""
    return 16000


# ---------------------------------------------------------------------------
# Dataclass factories
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_result_factory() -> Callable[..., AudioResult]:
    """Factory that builds an AudioResult with sensible defaults."""

    def _make(
        label: AudioClass = "chainsaw",
        confidence: float = 0.90,
        raw_scores: dict | None = None,
    ) -> AudioResult:
        if raw_scores is None:
            raw_scores = {
                "chainsaw": 0.0,
                "gunshot": 0.0,
                "engine": 0.0,
                "axe": 0.0,
                "fire": 0.0,
                "background": 0.0,
            }
            raw_scores[label] = confidence
        return AudioResult(label=label, confidence=confidence, raw_scores=raw_scores)

    return _make


@pytest.fixture
def triangulation_result_factory() -> Callable[..., TriangulationResult]:
    """Factory that builds a TriangulationResult with sensible defaults."""

    def _make(
        lat: float = 55.7510,
        lon: float = 37.6130,
        error_m: float = 5.0,
    ) -> TriangulationResult:
        return TriangulationResult(lat=lat, lon=lon, error_m=error_m)

    return _make


# ---------------------------------------------------------------------------
# Microphone geometry
# ---------------------------------------------------------------------------


@pytest.fixture
def hexagon_mics() -> list[MicPosition]:
    """Six mics in hexagonal pattern (~100m radius)."""
    center_lat, center_lon = 55.7514, 37.6138
    r_lat, r_lon = 0.000898, 0.001572
    return [
        MicPosition(
            lat=center_lat + r_lat * np.sin(np.radians(a)),
            lon=center_lon + r_lon * np.cos(np.radians(a)),
        )
        for a in [0, 60, 120, 180, 240, 300]
    ]


@pytest.fixture
def triangle_mics() -> list[MicPosition]:
    """Three microphones forming a roughly equilateral triangle (~100 m sides)."""
    mic_a = MicPosition(lat=55.7510, lon=37.6130)
    mic_b = MicPosition(lat=55.7510, lon=37.6146)  # ~100 m east
    mic_c = MicPosition(lat=55.7519, lon=37.6138)  # ~100 m north
    return [mic_a, mic_b, mic_c]
