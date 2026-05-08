"""Regression: drone_photo broadcast must survive send_pending failure.

Before fix in cloud/interface/main.py: asyncio.gather(drone_task, send_pending)
ran without return_exceptions=True. Any send_pending error (readonly DB,
TG API blip, network jitter) cancelled drone_task between drone_moving
and the drone_photo broadcast. UI showed every section EXCEPT ДРОН.

After fix: send_pending failure is logged and surfaced as alert_failed,
drone_photo always broadcasts when drone_task itself succeeds.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _real_main():
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F401

    return sys.modules[mod_name]


def _make_deps(*, send_pending_raises: BaseException | None = None):
    """Minimal deps dict for _run_demo. Optionally make send_pending fail."""
    MicPosition = types.SimpleNamespace
    mic_sim = AsyncMock()
    mic_sim.get_signals.return_value = (
        [[0.0] * 16000, [0.0] * 16000, [0.0] * 16000],
        ["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"],
    )

    onset = types.SimpleNamespace(triggered=True, energy_ratio=5.0)
    audio_result = types.SimpleNamespace(
        label="chainsaw",
        confidence=0.92,
        raw_scores={"chainsaw": 0.92, "background": 0.08},
    )
    location = types.SimpleNamespace(lat=57.37, lon=44.63, error_m=50.0)
    decision = types.SimpleNamespace(
        send_drone=True,
        send_lora=False,
        priority="P0",
        reason="chainsaw detected",
    )

    photo = types.SimpleNamespace(b64="AAAA", data=b"\x00")

    # AsyncMock can't model async generators (drone.fly_to). Build a plain
    # stub class that gives `async for` and awaitable methods what they need.
    class _StubDrone:
        async def takeoff(self):
            return None

        async def fly_to(self, lat, lon):
            # Empty async generator: no positions yielded → drone_task
            # proceeds straight to capture_photo + drone_photo broadcast.
            if False:
                yield None  # pragma: no cover

        async def capture_photo(self):
            return photo

        async def return_home(self):
            return None

    drone = _StubDrone()

    incident = types.SimpleNamespace(id=42)
    vision = types.SimpleNamespace(
        description="trees",
        has_human=False,
        has_fire=False,
        has_felling=True,
        has_machinery=False,
        is_threat=False,
        time_of_day="день",
        people_count=0,
        equipment_types=[],
        vegetation_damage="нет",
        damage_area_estimate="нет",
    )
    alert = types.SimpleNamespace(text="...", priority="P0", incident_id=42)

    if send_pending_raises is not None:
        send_pending = AsyncMock(side_effect=send_pending_raises)
    else:
        send_pending = AsyncMock(return_value=incident)

    return {
        "MicPosition": MicPosition,
        "MicSimulator": MagicMock(return_value=mic_sim),
        "detect_onset": MagicMock(return_value=onset),
        "classify": MagicMock(return_value=audio_result),
        "triangulate": MagicMock(return_value=location),
        "decide": MagicMock(return_value=decision),
        "SimulatedDrone": MagicMock(return_value=drone),
        "classify_photo": AsyncMock(return_value=vision),
        "compose_alert": AsyncMock(return_value=alert),
        "send_pending": send_pending,
        "send_confirmed": AsyncMock(),
        "get_online": MagicMock(return_value=[]),
    }


@pytest.mark.asyncio
async def test_drone_photo_emits_when_send_pending_fails(_real_main):
    """drone_photo broadcast survives RuntimeError from send_pending.

    This is the regression for «нажал БНЗ — все секции есть, кроме ДРОН».
    """
    events: list[dict] = []

    async def collect(msg):
        events.append(msg)

    deps = _make_deps(send_pending_raises=RuntimeError("readonly database"))

    with (
        patch.object(_real_main, "_import_demo_deps", return_value=deps),
        patch.object(_real_main, "broadcast", new=collect),
        patch.object(_real_main, "_available_memory_mb", return_value=10000.0),
        patch(
            "cloud.db.microphones.random_point_in_boundary",
            return_value=(57.37, 44.63),
        ),
        patch(
            "cloud.db.microphones.get_nearest_online",
            return_value=[],
        ),
    ):
        await _real_main._run_demo("chainsaw")

    event_names = [e.get("event") for e in events]
    assert "drone_photo" in event_names, (
        f"drone_photo must broadcast even when send_pending fails. "
        f"Got events: {event_names}"
    )
    assert "alert_failed" in event_names, (
        f"send_pending failure must surface as alert_failed event. "
        f"Got events: {event_names}"
    )


@pytest.mark.asyncio
async def test_drone_photo_emits_in_happy_path(_real_main):
    """Sanity: with all deps healthy, the full sequence still emits drone_photo."""
    events: list[dict] = []

    async def collect(msg):
        events.append(msg)

    deps = _make_deps(send_pending_raises=None)

    with (
        patch.object(_real_main, "_import_demo_deps", return_value=deps),
        patch.object(_real_main, "broadcast", new=collect),
        patch.object(_real_main, "_available_memory_mb", return_value=10000.0),
        patch(
            "cloud.db.microphones.random_point_in_boundary",
            return_value=(57.37, 44.63),
        ),
        patch(
            "cloud.db.microphones.get_nearest_online",
            return_value=[],
        ),
    ):
        await _real_main._run_demo("chainsaw")

    event_names = [e.get("event") for e in events]
    assert "drone_photo" in event_names
    assert "alert_failed" not in event_names
    assert "alert_sent" in event_names
