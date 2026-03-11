"""Tests: _run_demo() resilience to classifier failures.

Cloud container must stay alive even when edge.audio.classifier
cannot load (OOM, TF crash) or raises at runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def _real_main():
    """Ensure cloud.interface.main is the real module, not a MagicMock.

    Other tests (test_drone_bot, test_bot_workflow) replace this module
    in sys.modules with a MagicMock. We must reload the real one.
    """
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    # If it's a MagicMock or missing, force reimport
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811

    return sys.modules[mod_name]


class TestRunDemoResilience:
    """_run_demo() must not crash the process on classifier errors."""

    def test_run_demo_handles_classifier_import_error(self, _real_main):
        """_run_demo() logs and returns when classifier import fails."""
        _run_demo = _real_main._run_demo

        with patch.object(
            _real_main,
            "_import_demo_deps",
            side_effect=ImportError("No module named 'tensorflow'"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
                assert result is None
            finally:
                loop.close()

    def test_run_demo_handles_classifier_runtime_error(self, _real_main):
        """_run_demo() survives MemoryError / RuntimeError during classify()."""
        _run_demo = _real_main._run_demo

        with patch.object(
            _real_main,
            "_import_demo_deps",
            side_effect=MemoryError("out of memory"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
                assert result is None
            finally:
                loop.close()

    def test_health_endpoint_independent_of_classifier(self, _real_main):
        """/health returns 200 regardless of classifier state."""
        health = _real_main.health

        with patch.dict("sys.modules", {"tensorflow": None, "tensorflow_hub": None}):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(health())
                assert result == {"status": "ok"}
            finally:
                loop.close()
