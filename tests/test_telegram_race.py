"""RED-phase tests: race-condition bug in send_pending / send_confirmed.

Bug: `_mark_sent(chat_id)` is called BEFORE `await bot.send_message(...)` /
`bot.send_photo(...)`. If the send fails, the rate limiter still treats the
message as "sent" — so the next legitimate alert in the cooldown window will
be silently dropped.

These tests assert that on send failure, `_is_rate_limited` returns False
(i.e., `_last_sent[chat_id]` was NOT touched). They MUST fail against the
current implementation (RED).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from cloud.notify import telegram
from cloud.agent.decision import Alert


CHAT_ID = 999_001


@pytest.fixture(autouse=True)
def _clean_state():
    """Clear shared rate-limit state before and after each test."""
    telegram._last_sent.clear()
    yield
    telegram._last_sent.clear()


def _make_ranger(chat_id: int = CHAT_ID):
    """Minimal stand-in for cloud.db.rangers.Ranger used by the routers."""
    return types.SimpleNamespace(
        id=1,
        name="Test Ranger",
        chat_id=chat_id,
        active=True,
        current_lat=57.37,
        current_lon=44.63,
    )


@pytest.mark.asyncio
async def test_send_pending_does_not_mark_sent_on_failure():
    """send_pending must NOT update _last_sent when bot.send_message fails."""
    failing_bot = MagicMock()
    failing_bot.send_message = AsyncMock(side_effect=BadRequest("blocked"))

    with (
        patch("cloud.notify.telegram.Bot", return_value=failing_bot),
        patch(
            "cloud.notify.telegram.get_recent_nearby_incident",
            return_value=None,
        ),
        patch("cloud.notify.telegram._is_quiet_hours", return_value=False),
        patch(
            "cloud.notify.telegram.create_incident",
            return_value=types.SimpleNamespace(id=1, alert_message_ids={}),
        ),
        patch(
            "cloud.notify.telegram.get_nearest_rangers",
            return_value=[_make_ranger()],
        ),
        patch(
            "cloud.notify.telegram._get_target_chat_ids",
            return_value=[CHAT_ID],
        ),
    ):
        # send_pending swallows the exception inside its try/except — no raise.
        await telegram.send_pending(
            lat=57.37,
            lon=44.63,
            audio_class="chainsaw",
            reason="test",
            confidence=0.92,
        )

    # Failure happened → cooldown must NOT be active → retry must be allowed.
    assert not telegram._is_rate_limited(CHAT_ID, "alert"), (
        "Bug: _mark_sent ran before the failed send — retry is now blocked."
    )


@pytest.mark.asyncio
async def test_send_confirmed_does_not_mark_sent_on_failure():
    """send_confirmed must NOT update _last_sent when bot.send_photo fails."""
    failing_bot = MagicMock()
    failing_bot.send_photo = AsyncMock(side_effect=BadRequest("blocked"))
    failing_bot.send_message = AsyncMock(side_effect=BadRequest("blocked"))

    alert = Alert(
        text="confirmed alert",
        priority="ВЫСОКИЙ",
        lat=57.37,
        lon=44.63,
    )

    with (
        patch("cloud.notify.telegram.Bot", return_value=failing_bot),
        patch(
            "cloud.notify.telegram._get_target_chat_ids",
            return_value=[CHAT_ID],
        ),
    ):
        # incident=None → fallback path that actually calls bot.send_photo.
        await telegram.send_confirmed(alert, photo_bytes=b"\x00" * 16, incident=None)

    assert not telegram._is_rate_limited(CHAT_ID, "alert"), (
        "Bug: _mark_sent ran before the failed send — retry is now blocked."
    )
