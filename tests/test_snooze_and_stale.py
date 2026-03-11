"""P0 tests: snooze resend, stale cleanup, spatial dedup, quiet hours.

23 tests covering demo-critical features with zero existing coverage.
"""

import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "RANGERS_DB_PATH" not in os.environ:
    _tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    _tmp.close()
    os.environ.setdefault("RANGERS_DB_PATH", _tmp.name)

from cloud.db.incidents import (
    create_incident,
    get_incident,
    get_stale_incidents,
    get_recent_nearby_incident,
    clear_all_incidents,
    update_incident,
    assign_chat_to_incident,
    get_active_incident_for_chat,
)
from cloud.notify.bot_handlers import _snooze_resend, snooze_callback
from cloud.notify.bot_app import _cleanup_stale_incidents
from cloud.notify.telegram import _is_quiet_hours, _last_sent


@pytest.fixture(autouse=True)
def _clean_state():
    clear_all_incidents()
    _last_sent.clear()
    yield
    clear_all_incidents()
    _last_sent.clear()


def _make_callback_update(chat_id: int = 111, data: str = ""):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = data
    update.callback_query.message.chat_id = chat_id
    return update


# ===================================================================
# TestSnoozeResend (7 tests)
# ===================================================================


class TestSnoozeResend:
    """Test _snooze_resend job callback."""

    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.send_pending_to_chat", new_callable=AsyncMock)
    async def test_snooze_resend_sends_for_pending(self, mock_send):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        ctx = MagicMock()
        ctx.job.data = {"chat_id": 100, "incident_id": incident.id}

        await _snooze_resend(ctx)

        mock_send.assert_called_once_with(
            chat_id=100,
            lat=incident.lat,
            lon=incident.lon,
            audio_class="chainsaw",
            reason="Повторный алерт после snooze",
            confidence=0.85,
            gating_level="alert",
            is_demo=False,
        )

    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.send_pending_to_chat", new_callable=AsyncMock)
    async def test_snooze_resend_skips_accepted(self, mock_send):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        update_incident(
            incident.id,
            status="accepted",
            accepted_at=time.time(),
            accepted_by_chat_id=100,
            accepted_by_name="Test",
        )
        ctx = MagicMock()
        ctx.job.data = {"chat_id": 100, "incident_id": incident.id}

        await _snooze_resend(ctx)

        mock_send.assert_not_called()

    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.send_pending_to_chat", new_callable=AsyncMock)
    async def test_snooze_resend_skips_nonexistent(self, mock_send):
        ctx = MagicMock()
        ctx.job.data = {"chat_id": 100, "incident_id": "nonexistent-id"}

        await _snooze_resend(ctx)

        mock_send.assert_not_called()

    @pytest.mark.asyncio
    @patch("cloud.notify.bot_handlers.send_pending_to_chat", new_callable=AsyncMock)
    async def test_snooze_resend_preserves_is_demo(self, mock_send):
        incident = create_incident("gunshot", 57.3, 44.8, 0.90, "alert", is_demo=True)
        ctx = MagicMock()
        ctx.job.data = {"chat_id": 200, "incident_id": incident.id}

        await _snooze_resend(ctx)

        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["is_demo"] is True

    @pytest.mark.asyncio
    async def test_snooze_callback_schedules_job(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        update = _make_callback_update(chat_id=300, data=f"snooze:{incident.id}")
        ctx = MagicMock()
        ctx.job_queue = MagicMock()

        await snooze_callback(update, ctx)

        ctx.job_queue.run_once.assert_called_once()
        call_kwargs = ctx.job_queue.run_once.call_args
        assert call_kwargs.kwargs["when"] == 900
        assert call_kwargs.kwargs["data"]["chat_id"] == 300
        assert call_kwargs.kwargs["data"]["incident_id"] == incident.id

    @pytest.mark.asyncio
    async def test_snooze_callback_no_job_if_not_pending(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        update_incident(
            incident.id,
            status="accepted",
            accepted_at=time.time(),
            accepted_by_chat_id=400,
            accepted_by_name="Test",
        )
        update = _make_callback_update(chat_id=400, data=f"snooze:{incident.id}")
        ctx = MagicMock()
        ctx.job_queue = MagicMock()

        await snooze_callback(update, ctx)

        ctx.job_queue.run_once.assert_not_called()

    @pytest.mark.asyncio
    async def test_snooze_callback_no_incident_no_crash(self):
        update = _make_callback_update(chat_id=500, data="snooze:nonexistent-id")
        ctx = MagicMock()
        ctx.job_queue = MagicMock()

        await snooze_callback(update, ctx)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "?" in text
        ctx.job_queue.run_once.assert_not_called()


# ===================================================================
# TestStaleIncidents (7 tests)
# ===================================================================


class TestStaleIncidents:
    """Test get_stale_incidents and _cleanup_stale_incidents."""

    def test_stale_pending_over_30min(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        incident.created_at = time.time() - 1801

        stale = get_stale_incidents()
        assert len(stale) == 1
        assert stale[0].id == incident.id

    def test_fresh_pending_not_stale(self):
        create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")

        stale = get_stale_incidents()
        assert len(stale) == 0

    def test_stale_accepted_over_60min(self):
        incident = create_incident("gunshot", 57.3, 44.8, 0.90, "alert")
        incident.status = "accepted"
        incident.accepted_at = time.time() - 3601

        stale = get_stale_incidents()
        assert len(stale) == 1
        assert stale[0].id == incident.id

    def test_accepted_without_accepted_at_ignored(self):
        incident = create_incident("engine", 57.3, 44.8, 0.70, "verify")
        incident.status = "accepted"
        incident.accepted_at = None  # edge case!

        stale = get_stale_incidents()
        assert len(stale) == 0

    def test_on_site_and_resolved_not_stale(self):
        inc1 = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        inc1.status = "on_site"
        inc1.created_at = time.time() - 7200

        inc2 = create_incident("gunshot", 57.31, 44.81, 0.90, "alert")
        inc2.status = "resolved"
        inc2.created_at = time.time() - 7200

        inc3 = create_incident("engine", 57.32, 44.82, 0.70, "alert")
        inc3.status = "false_alarm"
        inc3.created_at = time.time() - 7200

        stale = get_stale_incidents()
        assert len(stale) == 0

    @pytest.mark.asyncio
    async def test_cleanup_sets_false_alarm(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        incident.created_at = time.time() - 1801

        await _cleanup_stale_incidents(MagicMock())

        updated = get_incident(incident.id)
        assert updated.status == "false_alarm"
        assert "Автозакрытие" in updated.resolution_details

    @pytest.mark.asyncio
    async def test_cleanup_clears_chat_mapping(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        incident.status = "accepted"
        incident.accepted_at = time.time() - 3601
        incident.accepted_by_chat_id = 999
        assign_chat_to_incident(999, incident.id)

        await _cleanup_stale_incidents(MagicMock())

        assert get_active_incident_for_chat(999) is None


# ===================================================================
# TestSpatialDedup (4 tests)
# ===================================================================


class TestSpatialDedup:
    """Test get_recent_nearby_incident spatial deduplication."""

    def test_nearby_recent_found(self):
        create_incident("chainsaw", 57.3000, 44.8000, 0.85, "alert")

        result = get_recent_nearby_incident(57.3009, 44.8000)
        assert result is not None

    def test_far_incident_not_found(self):
        create_incident("chainsaw", 57.3000, 44.8000, 0.85, "alert")

        result = get_recent_nearby_incident(57.3500, 44.8500)
        assert result is None

    def test_old_incident_not_found(self):
        incident = create_incident("chainsaw", 57.3000, 44.8000, 0.85, "alert")
        incident.created_at = time.time() - 301

        result = get_recent_nearby_incident(57.3009, 44.8000)
        assert result is None

    def test_resolved_incident_not_found(self):
        incident = create_incident("chainsaw", 57.3000, 44.8000, 0.85, "alert")
        incident.status = "resolved"

        result = get_recent_nearby_incident(57.3009, 44.8000)
        assert result is None


# ===================================================================
# TestQuietHours (5 tests)
# ===================================================================


class TestQuietHours:
    """Test _is_quiet_hours with mocked datetime."""

    def _patch_hour(self, monkeypatch, hour):
        """Patch datetime.now in telegram module to return a specific hour."""
        mock_dt = MagicMock()
        mock_dt.now.return_value.hour = hour
        monkeypatch.setattr("cloud.notify.telegram.datetime", mock_dt)
        monkeypatch.setattr("cloud.notify.telegram.QUIET_HOURS_START", 22)
        monkeypatch.setattr("cloud.notify.telegram.QUIET_HOURS_END", 6)

    def test_quiet_verify_at_23h(self, monkeypatch):
        self._patch_hour(monkeypatch, 23)
        assert _is_quiet_hours("verify") is True

    def test_alert_bypasses_at_23h(self, monkeypatch):
        self._patch_hour(monkeypatch, 23)
        assert _is_quiet_hours("alert") is False

    def test_not_quiet_at_18h(self, monkeypatch):
        self._patch_hour(monkeypatch, 18)
        assert _is_quiet_hours("verify") is False

    def test_boundary_at_06h(self, monkeypatch):
        self._patch_hour(monkeypatch, 6)
        assert _is_quiet_hours("log") is False  # < not <=

    def test_boundary_at_22h(self, monkeypatch):
        self._patch_hour(monkeypatch, 22)
        assert _is_quiet_hours("log") is True  # >= start
