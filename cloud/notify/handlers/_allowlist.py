"""Allowlist gate for Telegram bots — FAUN-37.

Application-level TypeHandler installed at group=-1 (runs BEFORE all
business handlers). Drops Updates from chat_ids not in env-configured
allowlist. Empty allowlist => allow all (back-compat for dev), but
WARNs at every parse so misconfiguration is auditable.
"""

import logging
import os

from telegram import Update
from telegram.ext import ApplicationHandlerStop, ContextTypes

logger = logging.getLogger(__name__)

_BLOCK_TEXT = "Доступ к боту ограничен. Обратитесь к администратору."


def _parse_chat_ids(env_var: str) -> set[int]:
    """Parse comma-separated ints from env. Robust to whitespace and garbage.

    - Empty/unset env: returns empty set (interpreted by gate as 'allow all').
    - Invalid tokens (e.g. 'abc', '--1', '1.5') are logged at WARNING and skipped
      — the gate never raises ValueError mid-update.
    - Env set but parses to empty (all-invalid) is logged as a louder WARNING
      so operators can detect misconfiguration that silently disables security.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return set()

    result: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.add(int(token))
        except ValueError:
            logger.warning("%s: invalid chat_id token %r — skipping", env_var, token)

    if not result:
        logger.warning(
            "%s set to %r but parsed to empty allowlist — "
            "bot will accept ALL chat_ids (allow-all mode)",
            env_var,
            raw,
        )
    return result


def make_allowlist_gate(env_var: str, bot_label: str):
    """Build a TypeHandler-compatible callback that gates Updates by chat_id.

    Returns: async callback (update, context) -> None.
    Behavior:
      - Updates without effective_chat: pass through (e.g. inline_query).
      - Empty allowlist: pass through (allow-all mode, warned at parse).
      - chat_id in allowlist: pass through (normal handlers will run).
      - chat_id not in allowlist: send block reply / answer callback,
        log warning, raise ApplicationHandlerStop to halt dispatch
        for this Update — no business handler executes.

    Install via:
        app.add_handler(
            TypeHandler(Update, make_allowlist_gate(...)), group=-1
        )
    """

    async def gate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if chat is None:
            return  # nothing to gate (inline_query, channel_post w/o chat, etc.)

        allowed = _parse_chat_ids(env_var)
        if not allowed:
            return  # allow-all (back-compat for dev / first-run)

        if chat.id in allowed:
            return  # allowed — let business handlers run

        # Blocked — respond appropriately and halt dispatch.
        try:
            if update.callback_query:
                await update.callback_query.answer("Доступ ограничен.", show_alert=True)
            elif update.message:
                await update.message.reply_text(_BLOCK_TEXT)
        except Exception:
            logger.exception(
                "%s: failed to send block reply to chat_id=%s",
                bot_label,
                chat.id,
            )

        logger.warning("%s: blocked chat_id=%s", bot_label, chat.id)
        raise ApplicationHandlerStop

    return gate
