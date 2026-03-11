"""Tests: _run_demo always sends pipeline_end, even on errors."""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _real_main():
    """Ensure cloud.interface.main is the real module, not a MagicMock.

    Other tests (test_drone_bot, test_bot_workflow) replace this module
    in sys.modules with a MagicMock. We must reload the real one.
    """
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811

    return sys.modules[mod_name]


def _make_deps_stub():
    """Return a minimal deps dict that _import_demo_deps would return."""
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
        send_drone=True, priority="P0", reason="chainsaw detected"
    )

    photo = types.SimpleNamespace(b64="AAAA", data=b"\x00")
    drone = AsyncMock()
    drone.fly_to.return_value = AsyncMock(__aiter__=AsyncMock(return_value=iter([])))
    drone.capture_photo.return_value = photo

    incident = types.SimpleNamespace(id=42)

    return {
        "MicPosition": MicPosition,
        "MicSimulator": MagicMock(return_value=mic_sim),
        "detect_onset": MagicMock(return_value=onset),
        "classify": MagicMock(return_value=audio_result),
        "triangulate": MagicMock(return_value=location),
        "decide": MagicMock(return_value=decision),
        "SimulatedDrone": MagicMock(return_value=drone),
        "classify_photo": AsyncMock(side_effect=RuntimeError("Gemma timeout")),
        "compose_alert": AsyncMock(),
        "send_pending": AsyncMock(return_value=incident),
        "send_confirmed": AsyncMock(),
        "get_online": MagicMock(return_value=[]),
    }


def _extract_broadcast_events(mock_broadcast: AsyncMock) -> list[dict]:
    """Extract all event dicts from broadcast calls."""
    return [call.args[0] for call in mock_broadcast.call_args_list]


# ---------- Test 1: crash mid-pipeline → pipeline_end sent ----------


@pytest.mark.asyncio
async def test_run_demo_always_sends_pipeline_end(_real_main):
    """If classify_photo raises, pipeline_end must still be broadcast."""
    deps = _make_deps_stub()
    _run_demo = _real_main._run_demo

    with (
        patch.object(_real_main, "_import_demo_deps", return_value=deps),
        patch.object(_real_main, "broadcast", new_callable=AsyncMock) as mock_bc,
        patch(
            "cloud.db.microphones.random_point_in_boundary",
            return_value=(57.37, 44.63),
        ),
        patch(
            "cloud.db.microphones.get_nearest_online",
            return_value=[],
        ),
    ):
        await _run_demo("chainsaw")

    events = _extract_broadcast_events(mock_bc)
    end_events = [e for e in events if e.get("event") == "pipeline_end"]
    assert len(end_events) == 1, f"Expected exactly 1 pipeline_end, got {end_events}"
    assert end_events[0]["reason"] == "error"


# ---------- Test 2: import failure → pipeline_end sent ----------


@pytest.mark.asyncio
async def test_run_demo_import_failure_sends_pipeline_end(_real_main):
    """If _import_demo_deps raises ImportError, pipeline_end must be sent."""
    _run_demo = _real_main._run_demo

    with (
        patch.object(
            _real_main,
            "_import_demo_deps",
            side_effect=ImportError("no tensorflow"),
        ),
        patch.object(_real_main, "broadcast", new_callable=AsyncMock) as mock_bc,
    ):
        await _run_demo("chainsaw")

    events = _extract_broadcast_events(mock_bc)
    end_events = [e for e in events if e.get("event") == "pipeline_end"]
    assert len(end_events) == 1, f"Expected pipeline_end, got {end_events}"
    assert end_events[0]["reason"] == "import_error"


# ---------- Test 3: vision_classified broadcast includes has_machinery ----------


def test_vision_broadcast_includes_has_machinery():
    """The vision_classified broadcast dict must include has_machinery field."""
    source_path = Path(__file__).parent.parent / "cloud" / "interface" / "main.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    found_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            if "event" in keys and "vision_classified" in [
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            ]:
                assert "has_machinery" in keys, (
                    "vision_classified broadcast is missing 'has_machinery' key"
                )
                found_count += 1

    assert found_count > 0, "Could not find vision_classified broadcast dict in source"
