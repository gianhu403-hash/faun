"""Characterization tests for bot_handlers.get_handlers().

Safety net for the bot_handlers.py → handlers/ split refactoring.
Verifies all 18 handlers are registered with correct types and patterns.
"""

from telegram.ext import (
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
)


def _get_handlers():
    """Lazy import to avoid module-level side effects."""
    from cloud.notify.bot_handlers import get_handlers

    return get_handlers()


class TestHandlerRegistry:
    def test_total_handler_count(self):
        handlers = _get_handlers()
        assert len(handlers) == 18, f"Expected 18 handlers, got {len(handlers)}"

    def test_command_handlers_registered(self):
        """All 7 commands must be registered."""
        handlers = _get_handlers()
        commands = set()
        for h in handlers:
            if isinstance(h, CommandHandler):
                commands.update(h.commands)
        expected = {"start", "status", "stop", "test", "help", "cancel", "rangers"}
        assert commands == expected, f"Missing commands: {expected - commands}"

    def test_callback_patterns_registered(self):
        """All 7 callback patterns must be registered."""
        handlers = _get_handlers()
        patterns = {
            h.pattern.pattern for h in handlers if isinstance(h, CallbackQueryHandler)
        }
        expected = {
            r"^district:",
            r"^accept:",
            r"^dispatch_drone:",
            r"^verdict:",
            r"^rag:",
            r"^confirm_reg:",
            r"^snooze:",
        }
        assert patterns == expected, f"Patterns mismatch: {patterns} vs {expected}"

    def test_message_handlers_count(self):
        """4 message handlers: VOICE, LOCATION, PHOTO, TEXT."""
        handlers = _get_handlers()
        msg_handlers = [h for h in handlers if isinstance(h, MessageHandler)]
        assert len(msg_handlers) == 4, (
            f"Expected 4 MessageHandlers, got {len(msg_handlers)}"
        )
