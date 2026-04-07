"""Telegram bot handlers for ranger self-registration and alerts.

Commands:
  /start  - Begin registration or show welcome if already registered
  /status - Show current registration details
  /stop   - Deactivate alerts (ranger remains in DB but active=False)
  /test   - Send a test alert with inline buttons for demo

Workflow callbacks:
  accept:<incident_id>              - Ranger accepts a call
  verdict:confirmed:<incident_id>   - Violation confirmed on site
  verdict:false:<incident_id>       - False alarm

Message handlers:
  PHOTO    - Ranger sends evidence photo (or standalone photo for Vision)
  VOICE    - Ranger sends voice description (STT -> text)
  LOCATION - Ranger shares location (proximity check)
"""

import base64
import logging
import os
import random
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from cloud.db.rangers import (
    add_ranger,
    get_ranger_by_chat_id,
    get_all_rangers,
    set_active,
    update_position,
)
from cloud.db.incidents import (
    get_incident,
    get_active_incident_for_chat,
    assign_chat_to_incident,
    clear_chat_incident,
    update_status,
    update_incident,
)
from cloud.notify.districts import DISTRICTS
from cloud.notify.telegram import (
    send_pending_to_chat,
    send_drone_photo,
    send_arrival_question,
    send_evidence_request,
    send_protocol_pdf,
    CLASS_NAME_RU,
    BOT_TOKEN,
)
from cloud.notify.handlers._shared import (
    ADMIN_CHAT_IDS,
    _registration_state,
    _REG_STEP_NAME,
    _REG_STEP_BADGE,
    _REG_STEP_CONFIRM,
    _REG_TTL,
    _haversine,
    _safe_answer,
)
from cloud.notify.handlers.commands import (
    status,
    stop,
    test_alert,
    help_cmd,
    rangers_cmd,
)
from cloud.notify.handlers.incident import (
    accept_callback,
    location_handler,
    verdict_callback,
    snooze_callback,
    _snooze_resend,
    dispatch_drone_callback,
)

logger = logging.getLogger(__name__)


# ---------- /start, /status, /stop ----------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register new ranger or greet existing one."""
    chat_id = update.effective_chat.id
    existing = get_ranger_by_chat_id(chat_id)

    if existing:
        if not existing.active:
            set_active(chat_id, True)
            await update.message.reply_text(
                f"С возвращением, {existing.name}! Оповещения снова включены."
            )
        else:
            await update.message.reply_text(
                f"Вы уже зарегистрированы, {existing.name}.\n"
                "Используйте /status для проверки или /stop для отключения."
            )
        return

    keyboard = [
        [InlineKeyboardButton(d.name_ru, callback_data=f"district:{slug}")]
        for slug, d in DISTRICTS.items()
    ]
    await update.message.reply_text(
        "Добро пожаловать в ForestGuard!\n\nВыберите ваше лесничество:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def district_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button press — register ranger for chosen district."""
    query = update.callback_query
    await _safe_answer(query)

    data = query.data
    if not data.startswith("district:"):
        return

    slug = data.split(":", 1)[1]
    district = DISTRICTS.get(slug)
    if not district:
        await query.edit_message_text("Неизвестное лесничество.")
        return

    chat_id = query.message.chat_id

    if get_ranger_by_chat_id(chat_id):
        await query.edit_message_text("Вы уже зарегистрированы! /status")
        return

    _registration_state[chat_id] = {
        "step": _REG_STEP_NAME,
        "district_slug": slug,
        "started_at": time.time(),
    }
    await query.edit_message_text(
        f"Лесничество: {district.name_ru}\n\n"
        "Шаг 1 из 3: Введите ваше ФИО (фамилия, имя, отчество):"
    )


# status, stop, test_alert — moved to cloud/notify/handlers/commands.py


# ---------- Accept callback ----------


# accept_callback, location_handler, verdict_callback —
# moved to cloud/notify/handlers/incident.py


# ---------- Voice handler ----------


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice message — STT and save as ranger report."""
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if not incident or incident.status != "on_site":
        await update.message.reply_text("Нет активного инцидента для записи.")
        return

    await update.message.reply_text("Распознаю голосовое сообщение...")

    try:
        voice_file = await update.message.voice.get_file()
        voice_bytes = await voice_file.download_as_bytearray()

        from cloud.agent.stt import recognize_voice

        text = await recognize_voice(bytes(voice_bytes))

        if not text:
            await update.message.reply_text(
                "Не удалось распознать голос. Опишите ситуацию текстом."
            )
            return

        incident.ranger_report_raw = text
        update_incident(incident.id, ranger_report_raw=text)
        logger.info(
            "AUDIT chat_id=%s action=evidence_voice incident=%s result=ok",
            chat_id,
            incident.id,
        )
        await update.message.reply_text(f'Текст сохранен:\n"{text}"')

        # If photo already collected, generate protocol
        if incident.ranger_photo_b64:
            await _generate_and_send_protocol(chat_id, incident)

    except ConnectionError:
        logger.exception("Voice handler: STT connection failed")
        await update.message.reply_text(
            "Сервис распознавания речи недоступен. Опишите ситуацию текстом."
        )
    except Exception as e:
        logger.exception("Voice handler failed: %s", type(e).__name__)
        await update.message.reply_text(
            "Ошибка обработки голосового сообщения. Попробуйте ещё раз или опишите текстом."
        )


# ---------- Photo handler ----------


async def handle_inspector_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle photo from inspector.

    If there's an active on_site incident, save as evidence.
    Otherwise, classify via YandexGPT Vision (standalone mode).
    """
    chat_id = update.effective_chat.id
    incident = get_active_incident_for_chat(chat_id)

    if incident and incident.status == "on_site":
        # Evidence collection mode
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            incident.ranger_photo_b64 = base64.b64encode(photo_bytes).decode()
            update_incident(incident.id, ranger_photo_b64=incident.ranger_photo_b64)

            # Check for caption as report text
            if update.message.caption:
                incident.ranger_report_raw = update.message.caption
                update_incident(incident.id, ranger_report_raw=update.message.caption)

            logger.info(
                "AUDIT chat_id=%s action=evidence_photo incident=%s result=ok",
                chat_id,
                incident.id,
            )
            await update.message.reply_text("Фото сохранено.")

            if incident.ranger_report_raw:
                await _generate_and_send_protocol(chat_id, incident)
            else:
                await update.message.reply_text(
                    "Опишите нарушение (текстом или голосовым сообщением)."
                )
        except ConnectionError:
            logger.exception("Evidence photo: download failed")
            await update.message.reply_text(
                "Не удалось скачать фото. Проверьте соединение и попробуйте снова."
            )
        except Exception as e:
            logger.exception("Evidence photo save failed: %s", type(e).__name__)
            await update.message.reply_text(
                "Ошибка сохранения фото. Попробуйте отправить ещё раз."
            )
        return

    # No active incident — photo analysis goes through the Drone bot
    await update.message.reply_text(
        "Нет активного инцидента. Отправьте фото через Drone-бот."
    )


# ---------- Text handler (for on_site report without voice) ----------


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text message — registration flow or ranger report if on_site."""
    chat_id = update.effective_chat.id
    text = update.message.text
    if not text or text.startswith("/"):
        return

    # --- Registration flow ---
    reg = _registration_state.get(chat_id)
    if reg:
        elapsed = time.time() - reg["started_at"]
        if elapsed > _REG_TTL:
            _registration_state.pop(chat_id, None)
            await update.message.reply_text(
                "Регистрация просрочена. Отправьте /start чтобы начать заново."
            )
            return

        if reg["step"] == _REG_STEP_NAME:
            name = text.strip()
            if len(name.split()) < 2:
                await update.message.reply_text(
                    "Введите полное ФИО (минимум фамилия и имя)."
                )
                return
            reg["name"] = name
            reg["step"] = _REG_STEP_BADGE
            await update.message.reply_text("Шаг 2 из 3: Введите ваш табельный номер:")
            return

        if reg["step"] == _REG_STEP_BADGE:
            badge = text.strip()
            if not badge:
                await update.message.reply_text("Табельный номер не может быть пустым.")
                return

            reg["badge"] = badge
            reg["step"] = _REG_STEP_CONFIRM

            slug = reg["district_slug"]
            district = DISTRICTS.get(slug)
            district_name = district.name_ru if district else slug

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Подтвердить", callback_data="confirm_reg:yes"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Начать заново", callback_data="confirm_reg:no"
                        )
                    ],
                ]
            )
            await update.message.reply_text(
                f"Шаг 3 из 3: Проверьте данные:\n\n"
                f"ФИО: {reg['name']}\n"
                f"Табельный номер: {badge}\n"
                f"Лесничество: {district_name}",
                reply_markup=keyboard,
            )
            return

        if reg["step"] == _REG_STEP_CONFIRM:
            # Waiting for button press, ignore text
            await update.message.reply_text(
                "Нажмите кнопку «Подтвердить» или «Начать заново»."
            )
            return

    # --- On-site report ---
    incident = get_active_incident_for_chat(chat_id)

    if not incident or incident.status != "on_site":
        ranger = get_ranger_by_chat_id(chat_id)
        if ranger:
            await update.message.reply_text(
                "Я не понял сообщение. Используйте /help для списка команд."
            )
        else:
            await update.message.reply_text(
                "Вы не зарегистрированы. Отправьте /start для начала."
            )
        return

    incident.ranger_report_raw = text
    update_incident(incident.id, ranger_report_raw=text)
    await update.message.reply_text("Описание сохранено.")

    if incident.ranger_photo_b64:
        await _generate_and_send_protocol(chat_id, incident)
    else:
        await update.message.reply_text("Отправьте фото нарушения.")


# ---------- RAG callback ----------


async def rag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle RAG inline button callbacks (rag:action:... or rag:protocol:...)."""
    query = update.callback_query
    await _safe_answer(query)

    data = query.data
    parts = data.split(":")
    if len(parts) < 5:
        await query.message.reply_text("Некорректные данные кнопки.")
        return

    _, rag_type, audio_class, lat_str, lon_str = parts[:5]
    lat = float(lat_str)
    lon = float(lon_str)
    class_ru = CLASS_NAME_RU.get(audio_class, audio_class)

    try:
        from cloud.agent.rag_agent import query_action, query_protocol

        await query.message.reply_text("Запрашиваю рекомендации...")

        if rag_type == "action":
            result = await query_action(audio_class, lat, lon)
        elif rag_type == "protocol":
            result = await query_protocol(audio_class, lat, lon)
        else:
            await query.message.reply_text("Неизвестный тип запроса.")
            return

        await query.message.reply_text(
            f"*{class_ru}* -- {rag_type}\n\n{result}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("RAG query failed")
        await query.message.reply_text(f"Ошибка RAG: {e}")


# ---------- Protocol generation ----------


async def _generate_and_send_protocol(chat_id: int, incident) -> None:
    """Generate legal text via YandexGPT, get legal articles via RAG, build PDF."""
    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=chat_id, text="Формирую протокол...")

    # 1. YandexGPT: raw report -> legal language
    try:
        from cloud.agent.rag_agent import legalize_report

        legal_text = await legalize_report(
            incident.audio_class, incident.ranger_report_raw
        )
        incident.ranger_report_legal = legal_text
        update_incident(incident.id, ranger_report_legal=legal_text)
    except Exception as e:
        logger.warning("Failed to legalize report via YandexGPT: %s", e)
        incident.ranger_report_legal = incident.ranger_report_raw
        update_incident(incident.id, ranger_report_legal=incident.ranger_report_raw)
        await bot.send_message(
            chat_id=chat_id,
            text="Не удалось обработать описание через YandexGPT, "
            "используется исходный текст.",
        )

    # 2. RAG: get applicable legal articles only
    legal_articles = ""
    try:
        from cloud.agent.rag_agent import query_legal_articles

        legal_articles = await query_legal_articles(
            incident.audio_class, incident.lat, incident.lon
        )
    except Exception as e:
        logger.warning("RAG query for legal articles failed: %s", e)

    # 3. Generate PDF
    try:
        from cloud.agent.protocol_pdf import generate_protocol

        pdf_bytes = generate_protocol(incident, legal_articles)
        incident.protocol_pdf = pdf_bytes
    except Exception as e:
        logger.exception("PDF generation failed")
        await bot.send_message(
            chat_id=chat_id,
            text="Ошибка генерации PDF-протокола.",
        )
        return

    # 4. Send PDF and resolve
    logger.info(
        "AUDIT chat_id=%s action=protocol_generated incident=%s result=ok",
        chat_id,
        incident.id,
    )
    await send_protocol_pdf(chat_id, pdf_bytes)
    update_incident(
        incident.id,
        status="resolved",
        resolution_details="Протокол составлен, материалы переданы",
    )
    incident.status = "resolved"
    clear_chat_incident(chat_id)


# ---------- /help ----------


# help_cmd — moved to cloud/notify/handlers/commands.py


# ---------- /cancel ----------


async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cancel — cancel ongoing registration."""
    chat_id = update.effective_chat.id
    if chat_id in _registration_state:
        _registration_state.pop(chat_id, None)
        await update.message.reply_text(
            "Регистрация отменена. Отправьте /start чтобы начать заново."
        )
    else:
        await update.message.reply_text("Нет активной регистрации для отмены.")


# ---------- /rangers (admin) ----------


# rangers_cmd — moved to cloud/notify/handlers/commands.py


# ---------- Registration confirmation callback ----------


async def confirm_reg_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle confirm_reg:yes / confirm_reg:no buttons."""
    query = update.callback_query
    await _safe_answer(query)

    chat_id = query.message.chat_id
    reg = _registration_state.get(chat_id)

    if not reg or reg["step"] != _REG_STEP_CONFIRM:
        await query.edit_message_text("Регистрация не найдена. Отправьте /start.")
        return

    answer = query.data.split(":", 1)[1]

    if answer == "no":
        _registration_state.pop(chat_id, None)
        await query.edit_message_text(
            "Регистрация отменена. Отправьте /start чтобы начать заново."
        )
        return

    # answer == "yes" — complete registration
    slug = reg["district_slug"]
    district = DISTRICTS.get(slug)
    if not district:
        _registration_state.pop(chat_id, None)
        await query.edit_message_text("Ошибка регистрации. Попробуйте /start.")
        return

    try:
        add_ranger(
            name=reg["name"],
            chat_id=chat_id,
            badge_number=reg["badge"],
            zone_lat_min=district.lat_min,
            zone_lat_max=district.lat_max,
            zone_lon_min=district.lon_min,
            zone_lon_max=district.lon_max,
        )
    except Exception:
        logger.exception("Failed to register ranger chat_id=%s", chat_id)
        _registration_state.pop(chat_id, None)
        await query.edit_message_text("Ошибка регистрации. Попробуйте позже.")
        return

    # Assign random position within the district
    rand_lat = round(random.uniform(district.lat_min, district.lat_max), 6)
    rand_lon = round(random.uniform(district.lon_min, district.lon_max), 6)
    update_position(chat_id, rand_lat, rand_lon)

    _registration_state.pop(chat_id, None)
    await query.edit_message_text(
        f"Вы зарегистрированы!\n\n"
        f"ФИО: {reg['name']}\n"
        f"Табельный номер: {reg['badge']}\n"
        f"Лесничество: {district.name_ru}\n"
        f"Ваша позиция: {rand_lat:.4f} N, {rand_lon:.4f} E\n\n"
        "Вы будете получать оповещения о подозрительной активности "
        "в вашей зоне. Используйте /stop для отключения."
    )


# ---------- Snooze callback ----------


# snooze_callback, _snooze_resend, dispatch_drone_callback —
# moved to cloud/notify/handlers/incident.py


# ---------- Handler registration ----------


def get_handlers() -> list:
    """Return all handlers to register on the Application."""
    return [
        CommandHandler("start", start),
        CommandHandler("status", status),
        CommandHandler("stop", stop),
        CommandHandler("test", test_alert),
        CommandHandler("help", help_cmd),
        CommandHandler("cancel", cancel_cmd),
        CommandHandler("rangers", rangers_cmd),
        CallbackQueryHandler(district_chosen, pattern=r"^district:"),
        CallbackQueryHandler(accept_callback, pattern=r"^accept:"),
        CallbackQueryHandler(dispatch_drone_callback, pattern=r"^dispatch_drone:"),
        CallbackQueryHandler(verdict_callback, pattern=r"^verdict:"),
        CallbackQueryHandler(rag_callback, pattern=r"^rag:"),
        CallbackQueryHandler(confirm_reg_callback, pattern=r"^confirm_reg:"),
        CallbackQueryHandler(snooze_callback, pattern=r"^snooze:"),
        MessageHandler(filters.VOICE, voice_handler),
        MessageHandler(filters.LOCATION, location_handler),
        MessageHandler(filters.PHOTO, handle_inspector_photo),
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler),
    ]
