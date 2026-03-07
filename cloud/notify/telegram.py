"""Telegram notifications — zone-based routing to rangers.

Instead of sending to a single hardcoded CHAT_ID, we query the ranger
database and send alerts ONLY to rangers whose zone covers the event
coordinates. If no rangers are found for a location, falls back to
TELEGRAM_CHAT_ID from env (admin/default channel).
"""

import os
import logging

from telegram import Bot
from telegram.constants import ParseMode
from cloud.agent.decision import Alert
from cloud.db.rangers import get_rangers_for_location, Ranger

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FALLBACK_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def _get_target_chat_ids(lat: float, lon: float) -> list[int]:
    """Get chat IDs of rangers responsible for this location."""
    rangers = get_rangers_for_location(lat, lon)
    if rangers:
        return [r.chat_id for r in rangers]
    # Fallback: if no rangers in DB or none cover this zone, use env default
    if FALLBACK_CHAT_ID:
        return [int(FALLBACK_CHAT_ID)]
    return []


async def send_pending(lat: float, lon: float, audio_class: str, reason: str) -> None:
    """Send initial alert to all rangers covering this location."""
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={lon},{lat}&z=15"

    text = (
        f"*Обнаружена аномалия*\n\n"
        f"Звук: `{audio_class}`\n"
        f"[{lat:.4f}°N, {lon:.4f}°E]({maps_link})\n\n"
        f"Дрон вылетел для подтверждения..."
    )

    chat_ids = _get_target_chat_ids(lat, lon)
    for chat_id in chat_ids:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.error("Failed to send pending alert to %s: %s", chat_id, e)


async def send_confirmed(alert: Alert, photo_bytes: bytes | None) -> None:
    """Send confirmed alert with photo to all rangers covering this location."""
    bot = Bot(token=BOT_TOKEN)
    maps_link = f"https://maps.yandex.ru/?pt={alert.lon},{alert.lat}&z=15"

    caption = (
        f"{alert.priority}\n\n"
        f"{alert.text}\n\n"
        f"[{alert.lat:.4f}°N, {alert.lon:.4f}°E]({maps_link})"
    )

    chat_ids = _get_target_chat_ids(alert.lat, alert.lon)
    for chat_id in chat_ids:
        try:
            if photo_bytes:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_bytes,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=caption,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True,
                )
        except Exception as e:
            logger.error("Failed to send confirmed alert to %s: %s", chat_id, e)
