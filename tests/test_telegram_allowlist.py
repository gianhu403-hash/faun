"""FAUN-37 — Telegram allowlist gate tests.

The allowlist is enforced at application level via a TypeHandler installed
at group=-1 (see cloud.notify.handlers._allowlist). These tests target the
gate factory and parser directly — that's where the security guarantee lives.

Behaviour expected:

- ``ALLOWED_RANGER_CHAT_IDS`` / ``ALLOWED_DRONE_CHAT_IDS`` (comma-separated
  ints) gate ALL Updates for the respective bot.
- Empty / unset env => empty set => all chat_ids allowed (back-compat for
  dev), but a WARNING is logged at every parse so misconfiguration is
  auditable.
- Invalid tokens (``abc``, ``--1``, ``1.5``) are skipped with a WARNING and
  never raise — PTB never silently drops an Update because of garbage env.
- Blocked chat_ids receive a Russian denial message containing the substring
  ``доступ`` or ``ограничен``, and dispatch is halted via
  ``ApplicationHandlerStop`` so no business handler runs.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.ext import ApplicationHandlerStop

# cloud.interface.main pulls FastAPI which isn't in the test env; stub it
# before importing anything from cloud.notify so module-level imports there
# don't explode at collection time.
import cloud.interface

_mock_main_module = MagicMock()
_mock_main_module.broadcast = AsyncMock()
sys.modules.setdefault("cloud.interface.main", _mock_main_module)
cloud.interface.main = _mock_main_module  # type: ignore[attr-defined]

from cloud.notify.handlers._allowlist import (
    _parse_chat_ids,
    make_allowlist_gate,
)


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_update(
    chat_id: int | None,
    *,
    with_message: bool = True,
    with_callback_query: bool = False,
):
    """Build a MagicMock that quacks like ``telegram.Update``."""
    update = MagicMock()
    if chat_id is None:
        update.effective_chat = None
    else:
        update.effective_chat.id = chat_id

    if with_message:
        update.message.reply_text = AsyncMock()
    else:
        update.message = None

    if with_callback_query:
        update.callback_query.answer = AsyncMock()
    else:
        update.callback_query = None

    return update


def _make_context():
    ctx = MagicMock()
    ctx.bot = MagicMock()
    ctx.bot.send_message = AsyncMock()
    return ctx


def _denial_in(reply_mock: AsyncMock) -> bool:
    """True iff any reply_text/answer call carried a Russian denial substring."""
    for call in reply_mock.call_args_list:
        text = (call.args[0] if call.args else call.kwargs.get("text", "")) or ""
        lower = text.lower()
        if "доступ" in lower or "ограничен" in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# _parse_chat_ids — env parsing
# ---------------------------------------------------------------------------


class TestParseChatIds:
    def test_parse_chat_ids_empty_env_returns_empty_set(self, monkeypatch):
        """Unset env => empty set (allow-all mode for the gate)."""
        monkeypatch.delenv("ALLOWED_TEST_IDS", raising=False)
        assert _parse_chat_ids("ALLOWED_TEST_IDS") == set()

    def test_parse_chat_ids_simple_list(self, monkeypatch):
        """env "1,2,3" => {1, 2, 3}."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1,2,3")
        assert _parse_chat_ids("ALLOWED_TEST_IDS") == {1, 2, 3}

    def test_parse_chat_ids_handles_whitespace(self, monkeypatch):
        """env "1, 2 ,3" => {1, 2, 3} — extra whitespace ignored."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1, 2 ,3")
        assert _parse_chat_ids("ALLOWED_TEST_IDS") == {1, 2, 3}

    def test_parse_chat_ids_handles_negative_chat_ids(self, monkeypatch):
        """Telegram group chat_ids are negative — must be parsed."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "-100123,42")
        assert _parse_chat_ids("ALLOWED_TEST_IDS") == {-100123, 42}

    def test_parse_chat_ids_skips_invalid_logs_warning(self, monkeypatch, caplog):
        """env "abc,1,--2,3" => {1, 3}, with WARNINGs for bad tokens."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "abc,1,--2,3")
        with caplog.at_level(logging.WARNING):
            result = _parse_chat_ids("ALLOWED_TEST_IDS")
        assert result == {1, 3}
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "'abc'" in joined or '"abc"' in joined
        assert "'--2'" in joined or '"--2"' in joined

    def test_parse_chat_ids_all_invalid_logs_louder_warning(self, monkeypatch, caplog):
        """env set but parses to empty => louder WARNING about allow-all mode."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "abc,xyz")
        with caplog.at_level(logging.WARNING):
            result = _parse_chat_ids("ALLOWED_TEST_IDS")
        assert result == set()
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "parsed to empty allowlist" in joined
        assert "allow-all" in joined.lower()


# ---------------------------------------------------------------------------
# make_allowlist_gate — runtime gate behaviour
# ---------------------------------------------------------------------------


class TestAllowlistGate:
    @pytest.mark.asyncio
    async def test_gate_allows_listed_chat_id(self, monkeypatch):
        """chat_id IN allowlist => gate returns None, no reply, no exception."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1,2,3")
        gate = make_allowlist_gate("ALLOWED_TEST_IDS", "test_bot")
        update = _make_update(chat_id=2)
        ctx = _make_context()

        # Must not raise.
        result = await gate(update, ctx)

        assert result is None
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_blocks_unlisted_chat_id_message(self, monkeypatch):
        """chat_id NOT in allowlist (text message) => denial reply + ApplicationHandlerStop."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1,2,3")
        gate = make_allowlist_gate("ALLOWED_TEST_IDS", "test_bot")
        update = _make_update(chat_id=999)
        ctx = _make_context()

        with pytest.raises(ApplicationHandlerStop):
            await gate(update, ctx)

        assert _denial_in(update.message.reply_text), (
            "Blocked chat_id must receive a denial message containing "
            "'доступ' or 'ограничен'"
        )

    @pytest.mark.asyncio
    async def test_gate_blocks_unlisted_chat_id_callback_query(self, monkeypatch):
        """chat_id NOT in allowlist (callback_query) => callback_query.answer + Stop."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1,2,3")
        gate = make_allowlist_gate("ALLOWED_TEST_IDS", "test_bot")
        update = _make_update(chat_id=999, with_message=False, with_callback_query=True)
        ctx = _make_context()

        with pytest.raises(ApplicationHandlerStop):
            await gate(update, ctx)

        assert _denial_in(update.callback_query.answer), (
            "Blocked callback_query must be answered with denial alert"
        )

    @pytest.mark.asyncio
    async def test_gate_empty_allowlist_passes_through_any_chat_id(self, monkeypatch):
        """Empty / unset env => allow-all mode, every chat_id passes."""
        monkeypatch.delenv("ALLOWED_TEST_IDS", raising=False)
        gate = make_allowlist_gate("ALLOWED_TEST_IDS", "test_bot")
        update = _make_update(chat_id=12345)
        ctx = _make_context()

        # Must not raise.
        result = await gate(update, ctx)

        assert result is None
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_no_effective_chat_passes_through(self, monkeypatch):
        """Update with no effective_chat (e.g. inline_query) => pass through."""
        monkeypatch.setenv("ALLOWED_TEST_IDS", "1,2,3")
        gate = make_allowlist_gate("ALLOWED_TEST_IDS", "test_bot")
        update = _make_update(chat_id=None, with_message=False)
        ctx = _make_context()

        # Must not raise — gate has nothing to gate.
        result = await gate(update, ctx)

        assert result is None
