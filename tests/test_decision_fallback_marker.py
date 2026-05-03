"""RED-phase test: silent fallback in `cloud/agent/decision.py` must be visible.

When YandexGPT call fails, `_call_yandex` returns a hardcoded string with no
marker, no priority change, no warning log, no metric. Operationally degraded
quality is invisible. This test encodes the expected behaviour.
"""

import logging
from unittest.mock import patch

import httpx
import pytest

from cloud.agent import decision


@pytest.mark.asyncio
async def test_yandexgpt_fallback_marks_alert_as_degraded(caplog):
    caplog.set_level(logging.WARNING)

    with patch(
        "cloud.agent.decision.httpx.AsyncClient.post",
        side_effect=httpx.TimeoutException("timeout"),
    ):
        alert = await decision.compose_alert(
            audio_class="chainsaw",
            visual_description="лес, просека",
            lat=57.3,
            lon=44.6,
            confidence=0.9,
        )

    assert alert.priority == "DEGRADED", (
        f"Fallback path must signal DEGRADED priority, got {alert.priority!r}"
    )
    assert alert.text.startswith("[fallback: AI недоступен]"), (
        f"Fallback text must carry visible marker, got {alert.text!r}"
    )
    assert any(
        rec.levelno == logging.WARNING and "METRIC alert_fallback_total" in rec.message
        for rec in caplog.records
    ), (
        "Expected a WARNING log line containing 'METRIC alert_fallback_total' "
        f"for ops grep-ability; got records: {[(r.levelname, r.message) for r in caplog.records]}"
    )
