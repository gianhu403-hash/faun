"""Structured logging for Faun — JSON log lines + job-scoped context.

Replaces the bare ``logger.exception("job %s failed", job_id)`` calls scattered
through ``faun.api`` with a single, structured logging setup. Log records become
one-line JSON objects (parseable by any log shipper) and a lightweight
``contextvars``-based filter injects the active ``job_id`` into every record
emitted within :func:`with_job_context` — so a job's failure carries its id as a
first-class field instead of being interpolated into the message.

Pure stdlib: no new dependency. ``setup_logging`` is idempotent — calling it
twice does not double-attach handlers — so it is safe to call at module import
and again at app startup.

Typical wiring:

    from faun.obs import setup_logging, with_job_context
    from faun.settings import get_settings

    setup_logging(json=get_settings().log_json)   # once, at app startup

    with with_job_context(job_id):
        ...                                         # records here carry job_id
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

#: Active job id for the current execution context (thread / async task safe).
#: ``None`` means "no job in scope" — the field is omitted from the record
#: rather than emitted as a noisy null.
_job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "faun_job_id", default=None
)

#: Marker attribute set on handlers we install, so :func:`setup_logging` can be
#: idempotent and ``_reset_for_tests`` can find them.
_HANDLER_MARK = "_faun_obs"

#: Standard ``LogRecord`` attributes — anything NOT in here is treated as a
#: caller-supplied ``extra`` field and merged into the JSON payload.
_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
        "job_id",
    }
)


class _JobContextFilter(logging.Filter):
    """Inject the context-local ``job_id`` onto every passing record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.job_id = _job_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single-line JSON object.

    Core fields are always present: ``timestamp`` (UTC, ISO-8601), ``level``,
    ``logger``, ``message``. ``job_id`` is included only when a job is in
    scope. Exceptions are serialized into an ``exc_info`` string. Any
    caller-supplied ``extra=`` keys are merged in (best-effort, str-coerced if
    not JSON-serializable).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        job_id = getattr(record, "job_id", None)
        if job_id is not None:
            payload["job_id"] = job_id

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(json: bool = True, *, level: int = logging.INFO) -> None:  # noqa: A002
    """Configure root logging once (idempotent).

    Installs a single stream handler on the root logger carrying either the
    :class:`_JsonFormatter` (``json=True``) or a plain human-readable formatter
    (``json=False``), plus the :class:`_JobContextFilter`. Re-invocation is a
    no-op for handler count — any previously installed Faun handler is replaced
    in place rather than stacked, so calling this at import and again at startup
    does not duplicate log lines.

    Args:
        json: Emit structured JSON (production) vs. plain text (local dev).
        level: Root logger level (default ``INFO``).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Drop any handler we previously installed (idempotency / mode switch).
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()  # defaults to stderr
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(_JobContextFilter())
    if json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)


@contextmanager
def with_job_context(job_id: str) -> Iterator[None]:
    """Bind ``job_id`` to every log record emitted within the ``with`` block.

    The id is stored in a :class:`contextvars.ContextVar`, so concurrent jobs
    (threads / async tasks) never see each other's id. The previous value is
    restored on exit — including when the guarded block raises — which makes
    nested contexts behave correctly and prevents id leakage after the job
    finishes.

    Args:
        job_id: The job identifier to attach to records.

    Yields:
        None. Use as ``with with_job_context(job_id): ...``.
    """
    token = _job_id_var.set(job_id)
    try:
        yield
    finally:
        _job_id_var.reset(token)


def _reset_for_tests() -> None:
    """Clear the context-local job id (test-only hygiene helper)."""
    _job_id_var.set(None)
