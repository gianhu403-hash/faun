"""Background task wrapper with strong reference + exception logging.

Raw asyncio.create_task() drops the task reference (GC may cancel it),
and unhandled exceptions only surface at GC time as
"Task exception was never retrieved" — invisible in prod logs.
This wrapper keeps a strong ref AND logs exceptions when they happen.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Coroutine, Any

logger = logging.getLogger(__name__)

_tasks: set[asyncio.Task] = set()


def safe_task(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        try:
            _tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "background task %r failed: %s",
                    name,
                    exc,
                    exc_info=exc,
                )
        except Exception:
            import sys

            print(f"safe_task _on_done failed for {name!r}", file=sys.stderr)

    task.add_done_callback(_on_done)
    return task
