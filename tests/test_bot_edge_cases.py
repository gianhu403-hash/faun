"""P1 tests: confirm_reg edge cases, error handlers, severity rate limiting,
accept/voice error paths.

12 tests covering reliability-critical edge cases.
"""

import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if "RANGERS_DB_PATH" not in os.environ:
    _tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    _tmp.close()
    os.environ.setdefault("RANGERS_DB_PATH", _tmp.name)

from cloud.db.rangers import init_db, get_ranger_by_chat_id
from cloud.db.incidents import (
    create_incident,
    clear_all_incidents,
    update_incident,
    assign_chat_to_incident,
)
from cloud.notify.bot_handlers import (
    confirm_reg_callback,
    voice_handler,
    accept_callback,
    _registration_state,
    _REG_STEP_CONFIRM,
)
from cloud.notify.bot_app import _error_handler
from cloud.notify.drone_bot_app import _error_handler as drone_error_handler
from cloud.notify.telegram import _is_rate_limited, _last_sent


@pytest.fixture(autouse=True)
def _clean_state():
    import sqlite3

    db_path = os.environ.get("RANGERS_DB_PATH", "")
    if db_path:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS rangers")
        conn.commit()
        conn.close()
    init_db()
    _registration_state.clear()
    clear_all_incidents()
    _last_sent.clear()
    yield
    _registration_state.clear()
    clear_all_incidents()
    _last_sent.clear()


def _make_callback_update(chat_id: int = 111, data: str = ""):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = data
    update.callback_query.message.chat_id = chat_id
    update.callback_query.from_user.full_name = "Test User"
    return update


# ===================================================================
# TestConfirmRegEdgeCases (3 tests)
# ===================================================================


class TestConfirmRegEdgeCases:
    @pytest.mark.asyncio
    async def test_confirm_no_reg_state(self):
        update = _make_callback_update(chat_id=3000, data="confirm_reg:yes")

        await confirm_reg_callback(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Регистрация не найдена" in text

    @pytest.mark.asyncio
    async def test_confirm_missing_district(self):
        _registration_state[3100] = {
            "step": _REG_STEP_CONFIRM,
            "district_slug": "nonexistent_district",
            "name": "Тест Тестов",
            "badge": "12345",
            "started_at": time.time(),
        }
        update = _make_callback_update(chat_id=3100, data="confirm_reg:yes")

        await confirm_reg_callback(update, MagicMock())

        assert 3100 not in _registration_state
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "ошибка" in text.lower()

    @pytest.mark.asyncio
    async def test_confirm_add_ranger_fails(self):
        _registration_state[3200] = {
            "step": _REG_STEP_CONFIRM,
            "district_slug": "varnavino",
            "name": "Тест Тестов",
            "badge": "12345",
            "started_at": time.time(),
        }
        update = _make_callback_update(chat_id=3200, data="confirm_reg:yes")

        with patch(
            "cloud.notify.bot_handlers.add_ranger", side_effect=Exception("DB error")
        ):
            await confirm_reg_callback(update, MagicMock())

        assert 3200 not in _registration_state
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "ошибка" in text.lower()
        assert get_ranger_by_chat_id(3200) is None


# ===================================================================
# TestErrorHandlers (4 tests)
# ===================================================================


class TestErrorHandlers:
    @pytest.mark.asyncio
    async def test_bot_error_sends_message(self):
        update = MagicMock()
        update.effective_chat.id = 100
        context = MagicMock()
        context.error = ValueError("test error")
        context.bot.send_message = AsyncMock()

        await _error_handler(update, context)

        context.bot.send_message.assert_called_once()
        assert context.bot.send_message.call_args.kwargs["chat_id"] == 100
        assert "ошибка" in context.bot.send_message.call_args.kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_bot_error_null_update(self):
        context = MagicMock()
        context.error = ValueError("test error")
        context.bot.send_message = AsyncMock()

        await _error_handler(None, context)

        context.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_error_send_fails_silently(self):
        update = MagicMock()
        update.effective_chat.id = 100
        context = MagicMock()
        context.error = ValueError("test error")
        context.bot.send_message = AsyncMock(side_effect=Exception("send failed"))

        # Should not raise
        await _error_handler(update, context)

    @pytest.mark.asyncio
    async def test_drone_error_sends_message(self):
        update = MagicMock()
        update.effective_chat.id = 200
        context = MagicMock()
        context.error = ValueError("drone error")
        context.bot.send_message = AsyncMock()

        await drone_error_handler(update, context)

        context.bot.send_message.assert_called_once()
        assert "ошибка" in context.bot.send_message.call_args.kwargs["text"].lower()


# ===================================================================
# TestSeverityRateLimiting (3 tests)
# ===================================================================


class TestSeverityRateLimiting:
    def test_verify_cooldown_300s(self):
        _last_sent[4001] = time.monotonic() - 1  # 1s ago
        assert _is_rate_limited(4001, "verify") is True

    def test_log_cooldown_600s(self):
        _last_sent[4002] = time.monotonic() - 301  # 301s ago, log cooldown is 600s
        assert _is_rate_limited(4002, "log") is True

    def test_alert_cooldown_61s_clear(self):
        _last_sent[4003] = time.monotonic() - 61  # 61s ago, alert cooldown is 60s
        assert _is_rate_limited(4003, "alert") is False


# ===================================================================
# TestAcceptAndErrorPaths (2 tests)
# ===================================================================


class TestAcceptAndErrorPaths:
    @pytest.mark.asyncio
    async def test_accept_location_failure_no_crash(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        update = _make_callback_update(chat_id=5001, data=f"accept:{incident.id}")
        ctx = MagicMock()

        mock_bot = MagicMock()
        mock_bot.send_location = AsyncMock(
            side_effect=Exception("location send failed")
        )
        mock_bot.send_message = AsyncMock()
        mock_bot.edit_message_text = AsyncMock()

        with patch("telegram.Bot", return_value=mock_bot):
            with patch(
                "cloud.notify.bot_handlers.send_drone_photo", new_callable=AsyncMock
            ):
                await accept_callback(update, ctx)

        # Function completed without crash; send_message was called after location failure
        mock_bot.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_voice_connection_error_message(self):
        incident = create_incident("chainsaw", 57.3, 44.8, 0.85, "alert")
        incident.status = "on_site"
        assign_chat_to_incident(5002, incident.id)

        update = MagicMock()
        update.effective_chat.id = 5002
        update.message.reply_text = AsyncMock()

        mock_voice_file = AsyncMock()
        mock_voice_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"fake-audio")
        )
        update.message.voice.get_file = AsyncMock(return_value=mock_voice_file)

        mock_stt = MagicMock()
        mock_stt.recognize_voice = AsyncMock(
            side_effect=ConnectionError("STT unavailable")
        )

        with patch.dict(sys.modules, {"cloud.agent.stt": mock_stt}):
            await voice_handler(update, MagicMock())

        replies = [call[0][0] for call in update.message.reply_text.call_args_list]
        assert any("недоступен" in r for r in replies)
