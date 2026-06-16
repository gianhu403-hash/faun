"""Shared fixtures for localization (TDOA triangulation) tests.

Carries the microphone-geometry fixtures the triangulation tests rely on,
ported from the v1 hackathon test suite. The AudioResult/yamnet factories are
intentionally NOT ported — the triangulation tests do not use them.
"""

from __future__ import annotations

import numpy as np
import pytest

from faun.localization.triangulate import MicPosition


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate used across the project."""
    return 16000


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
