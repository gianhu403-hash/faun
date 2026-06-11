"""Ranger registration handlers: /start, district selection, multi-step flow, /cancel."""

import logging
import random
import sqlite3
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from cloud.db.rangers import (
    add_ranger,
    get_ranger_by_chat_id,
    set_active,
    update_position,
)
from cloud.notify.districts import DISTRICTS
from cloud.notify.handlers._shared import (
    _REG_STEP_BADGE,
    _REG_STEP_CONFIRM,
    _REG_STEP_NAME,
    _REG_TTL,
    _registration_state,
    _safe_answer,
)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register new ranger or greet existing one.

    NOTE: chat_id allowlist enforcement is handled by the application-level
    TypeHandler installed in cloud.notify.bot_app.build_application
    (see cloud.notify.handlers._allowlist). FAUN-37.
    """
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
    except sqlite3.IntegrityError:
        logger.warning("Duplicate registration chat_id=%s", chat_id)
        _registration_state.pop(chat_id, None)
        await query.edit_message_text(
            "Вы уже зарегистрированы. Отправьте /status для проверки."
        )
        return
    except sqlite3.OperationalError:
        logger.exception("DB unavailable for chat_id=%s", chat_id)
        _registration_state.pop(chat_id, None)
        await query.edit_message_text(
            "База данных временно недоступна. Сообщите администратору."
        )
        return
    except Exception:
        logger.exception("Failed to register ranger chat_id=%s", chat_id)
        _registration_state.pop(chat_id, None)
        await query.edit_message_text("Ошибка регистрации. Попробуйте позже.")
        return

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


async def _handle_registration_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """Handle text in registration flow. Returns True if message was consumed."""
    chat_id = update.effective_chat.id
    text = update.message.text

    reg = _registration_state.get(chat_id)
    if not reg:
        return False

    elapsed = time.time() - reg["started_at"]
    if elapsed > _REG_TTL:
        _registration_state.pop(chat_id, None)
        await update.message.reply_text(
            "Регистрация просрочена. Отправьте /start чтобы начать заново."
        )
        return True

    if reg["step"] == _REG_STEP_NAME:
        name = text.strip()
        if len(name.split()) < 2:
            await update.message.reply_text(
                "Введите полное ФИО (минимум фамилия и имя)."
            )
            return True
        reg["name"] = name
        reg["step"] = _REG_STEP_BADGE
        await update.message.reply_text("Шаг 2 из 3: Введите ваш табельный номер:")
        return True

    if reg["step"] == _REG_STEP_BADGE:
        badge = text.strip()
        if not badge:
            await update.message.reply_text("Табельный номер не может быть пустым.")
            return True

        reg["badge"] = badge
        reg["step"] = _REG_STEP_CONFIRM

        slug = reg["district_slug"]
        district = DISTRICTS.get(slug)
        district_name = district.name_ru if district else slug

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Подтвердить", callback_data="confirm_reg:yes")],
                [InlineKeyboardButton("Начать заново", callback_data="confirm_reg:no")],
            ]
        )
        await update.message.reply_text(
            f"Шаг 3 из 3: Проверьте данные:\n\n"
            f"ФИО: {reg['name']}\n"
            f"Табельный номер: {badge}\n"
            f"Лесничество: {district_name}",
            reply_markup=keyboard,
        )
        return True

    if reg["step"] == _REG_STEP_CONFIRM:
        await update.message.reply_text(
            "Нажмите кнопку «Подтвердить» или «Начать заново»."
        )
        return True

    return False
