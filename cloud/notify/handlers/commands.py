"""Simple stateless command handlers: /status, /stop, /test, /help, /rangers."""

import random

from telegram import Update
from telegram.ext import ContextTypes

from cloud.db.rangers import get_all_rangers, get_ranger_by_chat_id, set_active
from cloud.notify.handlers._shared import ADMIN_CHAT_IDS
from cloud.notify.telegram import send_pending_to_chat


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show registration details."""
    chat_id = update.effective_chat.id
    ranger = get_ranger_by_chat_id(chat_id)

    if not ranger:
        await update.message.reply_text(
            "Вы не зарегистрированы. Отправьте /start для регистрации."
        )
        return

    state = "включены" if ranger.active else "отключены"
    badge_line = (
        f"Табельный номер: {ranger.badge_number}\n" if ranger.badge_number else ""
    )
    await update.message.reply_text(
        f"ФИО: {ranger.name}\n"
        f"{badge_line}"
        f"Зона: {ranger.zone_lat_min:.2f}--{ranger.zone_lat_max:.2f} N, "
        f"{ranger.zone_lon_min:.2f}--{ranger.zone_lon_max:.2f} E\n"
        f"Оповещения: {state}"
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — deactivate alerts."""
    chat_id = update.effective_chat.id
    ranger = get_ranger_by_chat_id(chat_id)

    if not ranger:
        await update.message.reply_text("Вы не зарегистрированы.")
        return
    if not ranger.active:
        await update.message.reply_text("Оповещения уже отключены.")
        return

    set_active(chat_id, False)
    await update.message.reply_text(
        "Оповещения отключены. Отправьте /start чтобы включить снова."
    )


async def test_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /test — send a demo alert with random Varnavino coordinates."""
    chat_id = update.effective_chat.id
    lat = round(random.uniform(57.05, 57.55), 6)
    lon = round(random.uniform(44.60, 45.40), 6)
    classes = ["chainsaw", "gunshot", "engine", "axe"]
    audio_class = random.choice(classes)
    confidence = round(random.uniform(0.65, 0.98), 2)

    await update.message.reply_text("Отправляю тестовый алерт...")
    await send_pending_to_chat(
        chat_id=chat_id,
        lat=lat,
        lon=lon,
        audio_class=audio_class,
        reason="Test alert",
        confidence=confidence,
        gating_level="alert",
        is_demo=True,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show bot usage instructions."""
    await update.message.reply_text(
        "ForestGuard — бот для лесных инспекторов\n\n"
        "Команды:\n"
        "/start — Регистрация или активация\n"
        "/status — Статус регистрации\n"
        "/stop — Отключить оповещения\n"
        "/test — Тестовый алерт\n"
        "/help — Эта справка\n"
        "/cancel — Отменить регистрацию\n"
        "/rangers — Список инспекторов (админ)\n\n"
        "При получении алерта:\n"
        "1. Нажмите «Принять вызов»\n"
        "2. Отправьте геолокацию на месте\n"
        "3. Подтвердите или опровергните нарушение\n"
        "4. Отправьте фото и описание\n"
        "5. Получите PDF-протокол"
    )


async def rangers_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /rangers — show all registered rangers (admin only)."""
    chat_id = update.effective_chat.id
    if ADMIN_CHAT_IDS and chat_id not in ADMIN_CHAT_IDS:
        await update.message.reply_text("Эта команда доступна только администраторам.")
        return

    rangers = get_all_rangers()
    if not rangers:
        await update.message.reply_text("Нет зарегистрированных инспекторов.")
        return

    lines = [f"Инспекторы ({len(rangers)}):"]
    for r in rangers:
        state = "вкл" if r.active else "выкл"
        lines.append(f"• {r.name} [{r.badge_number}] — {state}")
    await update.message.reply_text("\n".join(lines))
