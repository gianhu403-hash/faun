"""Tests for faun.health — the lightweight /healthz handler.

Эти тесты РЕАЛЬНО вызывают handler (TF-free, чистый stdlib): проверяют контракт
payload-а, флаг версии и деградацию readiness без исключений.
"""

from __future__ import annotations

import faun.health as health_mod
from faun.health import FAUN_VERSION, SERVICE_NAME, health


def test_health_returns_ok_with_writable_jobs_root(tmp_path, monkeypatch) -> None:
    """Happy path: каталог job-ов пишется -> status == "ok" и полный контракт."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    payload = health()

    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["service"] == SERVICE_NAME == "faun-api"
    assert payload["version"] == FAUN_VERSION
    assert isinstance(payload["version"], str) and payload["version"]
    assert payload["jobs_root_writable"] is True


def test_health_required_keys_present(tmp_path, monkeypatch) -> None:
    """Frozen-контракт: как минимум status/service/version всегда присутствуют."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    payload = health()
    for key in ("status", "service", "version"):
        assert key in payload


def test_health_degrades_without_raising(monkeypatch) -> None:
    """Если readiness-проверка падает — status == "degraded", БЕЗ исключения."""
    monkeypatch.setattr(health_mod, "_jobs_root_ok", lambda: False)
    payload = health()  # не должно бросать

    assert payload["status"] == "degraded"
    assert payload["jobs_root_writable"] is False
    # Базовый контракт сохраняется и в degraded.
    assert payload["service"] == "faun-api"
    assert payload["version"] == FAUN_VERSION


def test_jobs_root_ok_swallows_exceptions(monkeypatch) -> None:
    """_jobs_root_ok никогда не бросает — внутреннюю ошибку глотает в False."""

    def _boom(*_a, **_kw):
        raise OSError("disk on fire")

    # mkdir внутри проверки рвётся -> функция обязана вернуть False, не упасть.
    monkeypatch.setattr(health_mod.Path, "mkdir", _boom)
    assert health_mod._jobs_root_ok() is False
