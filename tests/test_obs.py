"""Tests for faun.obs — structured JSON logging + job-scoped log context.

Pure stdlib. These tests REALLY emit log records through the configured root
logger and parse the resulting line as JSON (no mock-only assertions): they
prove that ``setup_logging`` installs a single JSON handler idempotently, that
each emitted record is a parseable JSON object carrying the standard fields,
and that ``with_job_context`` injects ``job_id`` into records emitted inside
its scope and removes it again afterwards.
"""

from __future__ import annotations

import json
import logging

import pytest

import faun.obs as obs
from faun.obs import setup_logging, with_job_context


@pytest.fixture(autouse=True)
def _restore_root_logger() -> None:
    """Snapshot/restore root-logger handlers + level around each test.

    setup_logging mutates the root logger; without restoration the JSON handler
    would leak into sibling tests' captured output.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    obs._reset_for_tests()


def _emit_and_capture(capsys, *, logger_name: str = "faun.test") -> list[dict]:
    """Emit one record and return all JSON objects printed to stderr."""
    logging.getLogger(logger_name).warning("hello %s", "world")
    err = capsys.readouterr().err
    return [json.loads(line) for line in err.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------


def test_emits_parseable_json(capsys) -> None:
    """A configured logger emits one parseable JSON object with the core fields."""
    setup_logging(json=True)
    records = _emit_and_capture(capsys)

    assert len(records) == 1
    rec = records[0]
    assert rec["level"] == "WARNING"
    assert rec["logger"] == "faun.test"
    assert rec["message"] == "hello world"
    assert "timestamp" in rec
    # No job context active -> job_id is absent (not a noisy null).
    assert "job_id" not in rec


def test_exception_is_serialized(capsys) -> None:
    """logger.exception includes a JSON-safe ``exc_info`` string field."""
    setup_logging(json=True)
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("faun.test").exception("it failed")
    err = capsys.readouterr().err
    rec = json.loads(err.splitlines()[-1])
    assert rec["message"] == "it failed"
    assert "ValueError: boom" in rec["exc_info"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_setup_logging_is_idempotent(capsys) -> None:
    """Calling setup_logging twice must NOT double-attach handlers (no dup lines)."""
    setup_logging(json=True)
    setup_logging(json=True)

    root = logging.getLogger()
    json_handlers = [h for h in root.handlers if getattr(h, "_faun_obs", False)]
    assert len(json_handlers) == 1

    records = _emit_and_capture(capsys)
    assert len(records) == 1  # one line, not two


def test_plain_mode_is_not_json(capsys) -> None:
    """json=False keeps human-readable text (the line is not valid JSON)."""
    setup_logging(json=False)
    logging.getLogger("faun.test").warning("plain line")
    err = capsys.readouterr().err
    assert "plain line" in err
    with pytest.raises(json.JSONDecodeError):
        json.loads(err.splitlines()[-1])


# ---------------------------------------------------------------------------
# Job context
# ---------------------------------------------------------------------------


def test_with_job_context_injects_job_id(capsys) -> None:
    """Records emitted inside the context carry the job_id; outside they do not."""
    setup_logging(json=True)

    with with_job_context("job-abc"):
        logging.getLogger("faun.test").warning("inside")
    logging.getLogger("faun.test").warning("outside")

    err = capsys.readouterr().err
    lines = [json.loads(line) for line in err.splitlines() if line.strip()]
    inside = next(r for r in lines if r["message"] == "inside")
    outside = next(r for r in lines if r["message"] == "outside")

    assert inside["job_id"] == "job-abc"
    assert "job_id" not in outside


def test_with_job_context_restores_on_exception(capsys) -> None:
    """The job_id is cleared even when the guarded block raises."""
    setup_logging(json=True)

    with pytest.raises(RuntimeError):
        with with_job_context("job-xyz"):
            raise RuntimeError("during job")

    logging.getLogger("faun.test").warning("after")
    err = capsys.readouterr().err
    rec = json.loads(err.splitlines()[-1])
    assert rec["message"] == "after"
    assert "job_id" not in rec


def test_nested_job_context_restores_outer(capsys) -> None:
    """Nested contexts restore the previous job_id (not just clear to none)."""
    setup_logging(json=True)

    with with_job_context("outer"):
        with with_job_context("inner"):
            logging.getLogger("faun.test").warning("nested")
        logging.getLogger("faun.test").warning("back-to-outer")

    err = capsys.readouterr().err
    lines = [json.loads(line) for line in err.splitlines() if line.strip()]
    nested = next(r for r in lines if r["message"] == "nested")
    outer = next(r for r in lines if r["message"] == "back-to-outer")
    assert nested["job_id"] == "inner"
    assert outer["job_id"] == "outer"
