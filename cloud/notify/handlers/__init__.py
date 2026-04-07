"""Bot handlers package — split from monolithic bot_handlers.py.

Public API:
- get_handlers() — returns the list of telegram.ext handlers for Application
- text_handler — composite dispatcher (registration → on-site → fallback)
"""

from telegram import Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from cloud.notify.handlers.alerts import fallback_text_reply, rag_callback
from cloud.notify.handlers.commands import (
    help_cmd,
    rangers_cmd,
    status,
    stop,
    test_alert,
)
from cloud.notify.handlers.evidence import (
    _handle_onsite_text,
    handle_inspector_photo,
    voice_handler,
)
from cloud.notify.handlers.incident import (
    accept_callback,
    dispatch_drone_callback,
    location_handler,
    snooze_callback,
    verdict_callback,
)
from cloud.notify.handlers.registration import (
    _handle_registration_text,
    cancel_cmd,
    confirm_reg_callback,
    district_chosen,
    start,
)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Composite text dispatcher: registration → on-site report → fallback."""
    text = update.message.text
    if not text or text.startswith("/"):
        return

    if await _handle_registration_text(update, context):
        return

    if await _handle_onsite_text(update, context):
        return

    await fallback_text_reply(update)


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
