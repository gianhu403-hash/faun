"""Incident lifecycle handlers: accept → location → verdict → snooze → drone dispatch."""

import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from cloud.db.incidents import (
    clear_chat_incident,
    get_active_incident_for_chat,
    get_incident,
    assign_chat_to_incident,
    update_incident,
)
from cloud.db.rangers import get_ranger_by_chat_id
from cloud.notify.handlers._shared import _haversine, _safe_answer
from cloud.notify.telegram import (
    BOT_TOKEN,
    CLASS_NAME_RU,
    send_drone_photo,
    send_pending_to_chat,
)

logger = logging.getLogger(__name__)


async def accept_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle 'Принять вызов' button."""
    query = update.callback_query
    await _safe_answer(query)

    parts = query.data.split(":", 1)
    if len(parts) < 2:
        return

    incident_id = parts[1]
    incident = get_incident(incident_id)

    if not incident:
        await query.edit_message_text("Инцидент не найден.")
        return

    if incident.status != "pending":
        await query.edit_message_text(
            f"Вызов уже принят: {incident.accepted_by_name or 'другой инспектор'}."
        )
        return

    chat_id = query.message.chat_id
    ranger = get_ranger_by_chat_id(chat_id)
    name = ranger.name if ranger else (query.from_user.full_name or str(chat_id))

    now = time.time()
    update_incident(
        incident_id,
        status="accepted",
        accepted_by_chat_id=chat_id,
        accepted_by_name=name,
        accepted_at=now,
    )
    incident.status = "accepted"
    incident.accepted_by_chat_id = chat_id
    incident.accepted_by_name = name
    incident.accepted_at = now
    assign_chat_to_incident(chat_id, incident_id)

    maps_url = f"https://maps.yandex.ru/?pt={incident.lon},{incident.lat}&z=15"

    logger.info(
        "AUDIT chat_id=%s action=accept incident=%s result=ok",
        chat_id,
        incident_id,
    )

    await query.edit_message_text(f"Вызов принят. Выезжайте на точку:\n{maps_url}")

    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)
    try:
        await bot.send_location(
            chat_id=chat_id, latitude=incident.lat, longitude=incident.lon
        )
    except Exception as e:
        logger.warning("Failed to send location to %s: %s", chat_id, e)

    for other_chat_id, msg_id in incident.alert_message_ids.items():
        if other_chat_id == chat_id:
            continue
        try:
            class_ru = CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
            await bot.edit_message_text(
                chat_id=other_chat_id,
                message_id=msg_id,
                text=(f"*АЛЕРТ: {class_ru}*\n━━━━━━━━━━━━━━━━\nВызов принял: {name}"),
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning("Failed to edit alert for chat %s: %s", other_chat_id, e)

    await send_drone_photo(chat_id, incident)

    await bot.send_message(
        chat_id=chat_id,
        text="Отправьте геолокацию, когда будете рядом с точкой.",
    )


async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle shared location — check proximity to incident."""
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if not incident:
        await update.message.reply_text("Нет активных вызовов.")
        return

    if incident.status not in ("accepted",):
        return

    loc = update.message.location
    dist = _haversine(loc.latitude, loc.longitude, incident.lat, incident.lon)

    PROXIMITY_RADIUS_M = 1000

    if incident.is_demo or dist <= PROXIMITY_RADIUS_M:
        now = time.time()
        resp_min = None
        if incident.created_at:
            resp_min = round((now - incident.created_at) / 60, 1)
        update_incident(
            incident.id,
            status="on_site",
            arrived_at=now,
            response_time_min=resp_min,
        )
        incident.status = "on_site"
        incident.arrived_at = now
        incident.response_time_min = resp_min
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Нарушение подтверждено",
                        callback_data=f"verdict:confirmed:{incident.id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Ложный вызов",
                        callback_data=f"verdict:false:{incident.id}",
                    ),
                ],
            ]
        )
        await update.message.reply_text(
            "Вы рядом с точкой. Что на месте?", reply_markup=keyboard
        )
    else:
        await update.message.reply_text(
            f"Вы в {dist:.0f} м от точки. Продолжайте движение."
        )


async def verdict_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle 'Нарушение подтверждено' / 'Ложный вызов' buttons."""
    query = update.callback_query
    await _safe_answer(query)

    parts = query.data.split(":")
    if len(parts) < 3:
        return

    _, verdict_type, incident_id = parts[:3]
    incident = get_incident(incident_id)

    if not incident:
        await query.edit_message_text("Инцидент не найден.")
        return

    chat_id = query.message.chat_id

    if verdict_type == "false":
        update_incident(
            incident_id,
            status="false_alarm",
            resolution_details="Ложное срабатывание, закрыто инспектором",
        )
        if incident:
            incident.status = "false_alarm"
        clear_chat_incident(chat_id)
        logger.info(
            "AUDIT chat_id=%s action=verdict:false incident=%s result=false_alarm",
            chat_id,
            incident_id,
        )
        await query.edit_message_text("Принято, инцидент закрыт. Спасибо за проверку.")

    elif verdict_type == "confirmed":
        logger.info(
            "AUDIT chat_id=%s action=verdict:confirmed incident=%s result=confirmed",
            chat_id,
            incident_id,
        )
        await query.edit_message_text("Нарушение зафиксировано.")
        await query.message.reply_text(
            "Пришлите фото нарушения и опишите ситуацию (текстом или голосовым сообщением)."
        )


async def snooze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle 'Отложить 15 мин' button — snooze alert and re-send later."""
    query = update.callback_query
    await _safe_answer(query)

    parts = query.data.split(":", 1)
    if len(parts) < 2:
        return

    incident_id = parts[1]
    chat_id = query.message.chat_id

    logger.info(
        "AUDIT chat_id=%s action=snooze incident=%s result=snoozed_15m",
        chat_id,
        incident_id,
    )

    incident = get_incident(incident_id)
    class_ru = (
        CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
        if incident
        else "?"
    )

    await query.edit_message_text(
        f"*АЛЕРТ: {class_ru}*\n━━━━━━━━━━━━━━━━\nОтложено на 15 минут",
        parse_mode="Markdown",
    )

    if context.job_queue and incident and incident.status == "pending":
        context.job_queue.run_once(
            _snooze_resend,
            when=900,
            data={"chat_id": chat_id, "incident_id": incident_id},
        )


async def _snooze_resend(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job callback: re-send alert after snooze expires."""
    data = context.job.data
    chat_id = data["chat_id"]
    incident_id = data["incident_id"]

    incident = get_incident(incident_id)
    if not incident or incident.status != "pending":
        return

    await send_pending_to_chat(
        chat_id=chat_id,
        lat=incident.lat,
        lon=incident.lon,
        audio_class=incident.audio_class,
        reason="Повторный алерт после snooze",
        confidence=incident.confidence,
        gating_level=incident.gating_level,
        is_demo=incident.is_demo,
    )


async def dispatch_drone_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle 'Отправить дрона' button for VERIFY-level alerts."""
    query = update.callback_query
    await _safe_answer(query)

    parts = query.data.split(":", 1)
    if len(parts) < 2:
        return

    incident_id = parts[1]
    chat_id = query.message.chat_id
    incident = get_incident(incident_id)

    if not incident:
        await query.edit_message_text("Инцидент не найден.")
        return

    if incident.status not in ("pending", "verify"):
        await query.edit_message_text(
            f"Инцидент уже обработан (статус: {incident.status})."
        )
        return

    logger.info(
        "AUDIT chat_id=%s action=dispatch_drone incident=%s",
        chat_id,
        incident_id,
    )

    class_ru = CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
    maps_url = f"https://maps.yandex.ru/?pt={incident.lon},{incident.lat}&z=15"

    await query.edit_message_text(
        f"*АЛЕРТ: {class_ru}*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Дрон отправлен на {incident.lat:.4f}°N {incident.lon:.4f}°E\n"
        f"Ожидайте фотографию\\.\\.\\.\n\n"
        f"[На карте]({maps_url})",
        parse_mode="MarkdownV2",
    )

    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:8000/api/v1/incidents/{incident_id}/dispatch-drone",
                timeout=10,
            )
            if resp.status_code != 200:
                logger.error(
                    "Drone dispatch API returned %s: %s",
                    resp.status_code,
                    resp.text,
                )
    except Exception as e:
        logger.error("Drone dispatch API call failed: %s", e)
