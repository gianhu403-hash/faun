"""Evidence collection handlers: voice, photo, protocol generation."""

import base64
import logging

from telegram import Update
from telegram.ext import ContextTypes

from cloud.db.incidents import (
    clear_chat_incident,
    get_active_incident_for_chat,
    update_incident,
)
from cloud.notify.telegram import BOT_TOKEN, send_protocol_pdf

logger = logging.getLogger(__name__)


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
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            incident.ranger_photo_b64 = base64.b64encode(photo_bytes).decode()
            update_incident(incident.id, ranger_photo_b64=incident.ranger_photo_b64)

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

    await update.message.reply_text(
        "Нет активного инцидента. Отправьте фото через Drone-бот."
    )


async def _generate_and_send_protocol(chat_id: int, incident) -> None:
    """Generate legal text via YandexGPT, get legal articles via RAG, build PDF."""
    from telegram import Bot

    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=chat_id, text="Формирую протокол...")

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

    legal_articles = ""
    try:
        from cloud.agent.rag_agent import query_legal_articles

        legal_articles = await query_legal_articles(
            incident.audio_class, incident.lat, incident.lon
        )
    except Exception as e:
        logger.warning("RAG query for legal articles failed: %s", e)

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
