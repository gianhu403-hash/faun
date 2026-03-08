"""Tests for Telegram bot registration handlers.

Uses mock Update/Message objects to test handler logic without
connecting to Telegram API.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
_tmp.close()
os.environ["RANGERS_DB_PATH"] = _tmp.name

from cloud.db.rangers import init_db, get_ranger_by_chat_id, add_ranger, set_active
from cloud.notify.bot_handlers import start, district_chosen, status, stop
from cloud.notify.districts import DISTRICTS


@pytest.fixture(autouse=True)
def _clean_db():
    import sqlite3
    conn = sqlite3.connect(os.environ["RANGERS_DB_PATH"])
    conn.execute("DROP TABLE IF EXISTS rangers")
    conn.commit()
    conn.close()
    init_db()
    yield


def _make_update(chat_id: int = 111, full_name: str = "Тест Тестов"):
    """Create a mock Update with message."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.chat_id = chat_id
    # For callback queries
    update.callback_query = None
    # User info
    user = MagicMock()
    user.full_name = full_name
    user.username = "test_user"
    update.message.from_user = user
    return update


def _make_callback_update(chat_id: int = 111, data: str = "district:varnavino",
                           full_name: str = "Тест Тестов"):
    """Create a mock Update with callback query."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = data
    update.callback_query.message.chat_id = chat_id
    update.callback_query.from_user.full_name = full_name
    update.callback_query.from_user.username = "test_user"
    return update


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_new_user_gets_district_keyboard(self):
        update = _make_update(chat_id=100)
        await start(update, MagicMock())

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        assert "Выберите ваше лесничество" in call_args[0][0]
        assert call_args[1]["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_existing_active_user_gets_greeting(self):
        add_ranger("Иван", chat_id=200, zone_lat_min=57.0, zone_lat_max=58.0,
                    zone_lon_min=44.0, zone_lon_max=46.0)
        update = _make_update(chat_id=200)
        await start(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "уже зарегистрированы" in text

    @pytest.mark.asyncio
    async def test_inactive_user_reactivated(self):
        add_ranger("Пётр", chat_id=300, zone_lat_min=57.0, zone_lat_max=58.0,
                    zone_lon_min=44.0, zone_lon_max=46.0)
        set_active(300, False)

        update = _make_update(chat_id=300)
        await start(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "С возвращением" in text
        assert get_ranger_by_chat_id(300).active is True


class TestDistrictChosen:
    @pytest.mark.asyncio
    async def test_registers_ranger_for_varnavino(self):
        update = _make_callback_update(chat_id=400, data="district:varnavino")
        await district_chosen(update, MagicMock())

        ranger = get_ranger_by_chat_id(400)
        assert ranger is not None
        assert ranger.name == "Тест Тестов"
        district = DISTRICTS["varnavino"]
        assert ranger.zone_lat_min == district.lat_min
        assert ranger.zone_lon_max == district.lon_max

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Вы зарегистрированы" in text
        assert "Варнавинское" in text

    @pytest.mark.asyncio
    async def test_unknown_district_shows_error(self):
        update = _make_callback_update(chat_id=500, data="district:unknown")
        await district_chosen(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "неизвестное лесничество" in text
        assert get_ranger_by_chat_id(500) is None

    @pytest.mark.asyncio
    async def test_already_registered_blocked(self):
        add_ranger("Уже есть", chat_id=600, zone_lat_min=57.0, zone_lat_max=58.0,
                    zone_lon_min=44.0, zone_lon_max=46.0)
        update = _make_callback_update(chat_id=600, data="district:varnavino")
        await district_chosen(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "уже зарегистрированы" in text


class TestStatusHandler:
    @pytest.mark.asyncio
    async def test_registered_user_sees_info(self):
        add_ranger("Сергей", chat_id=700, zone_lat_min=57.05, zone_lat_max=57.55,
                    zone_lon_min=44.60, zone_lon_max=45.40)
        update = _make_update(chat_id=700)
        await status(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "Сергей" in text
        assert "включены" in text

    @pytest.mark.asyncio
    async def test_unregistered_user_gets_prompt(self):
        update = _make_update(chat_id=800)
        await status(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "не зарегистрированы" in text


class TestStopHandler:
    @pytest.mark.asyncio
    async def test_active_ranger_deactivated(self):
        add_ranger("Алексей", chat_id=900, zone_lat_min=57.0, zone_lat_max=58.0,
                    zone_lon_min=44.0, zone_lon_max=46.0)
        update = _make_update(chat_id=900)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "отключены" in text
        assert get_ranger_by_chat_id(900).active is False

    @pytest.mark.asyncio
    async def test_already_inactive_ranger(self):
        add_ranger("Борис", chat_id=1000, zone_lat_min=57.0, zone_lat_max=58.0,
                    zone_lon_min=44.0, zone_lon_max=46.0)
        set_active(1000, False)
        update = _make_update(chat_id=1000)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "уже отключены" in text

    @pytest.mark.asyncio
    async def test_unregistered_user(self):
        update = _make_update(chat_id=1100)
        await stop(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "не зарегистрированы" in text
