"""Alert-related callback handlers (RAG queries from inline buttons)."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from cloud.db.rangers import get_ranger_by_chat_id
from cloud.notify.handlers._shared import _safe_answer
from cloud.notify.telegram import CLASS_NAME_RU

logger = logging.getLogger(__name__)


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


async def fallback_text_reply(update: Update) -> None:
    """Fallback text reply when no registration or on-site context applies."""
    chat_id = update.effective_chat.id
    ranger = get_ranger_by_chat_id(chat_id)
    if ranger:
        await update.message.reply_text(
            "Я не понял сообщение. Используйте /help для списка команд."
        )
    else:
        await update.message.reply_text(
            "Вы не зарегистрированы. Отправьте /start для начала."
        )
