import asyncio
import logging

import pytest

from cloud.interface._tasks import _tasks, safe_task


@pytest.mark.asyncio
async def test_safe_task_returns_asyncio_task() -> None:
    async def some_coro() -> None:
        return None

    task = safe_task(some_coro(), name="x")
    assert isinstance(task, asyncio.Task)
    await task


@pytest.mark.asyncio
async def test_safe_task_holds_strong_reference() -> None:
    async def some_coro() -> None:
        await asyncio.sleep(0)

    task = safe_task(some_coro(), name="x")
    assert task in _tasks

    await task
    await asyncio.sleep(0)

    assert task not in _tasks


@pytest.mark.asyncio
async def test_failing_task_logs_exception(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="cloud.interface._tasks")

    async def boom() -> None:
        raise RuntimeError("kaboom")

    task = safe_task(boom(), name="boom_task")

    try:
        await task
    except RuntimeError:
        pass

    await asyncio.sleep(0)

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error_records, "expected at least one ERROR-level log record"

    matching = [r for r in error_records if "boom_task" in r.getMessage()]
    assert matching, (
        f"no ERROR record mentions 'boom_task'; got: {[r.getMessage() for r in error_records]}"
    )

    rec = matching[0]
    assert rec.exc_info is not None, "logger.exception() should attach exc_info"
    exc_type, exc_value, _tb = rec.exc_info
    assert exc_type is RuntimeError
    assert "kaboom" in str(exc_value)
