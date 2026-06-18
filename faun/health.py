"""Лёгкий health-handler для liveness/readiness Faun-API.

Чистый stdlib: НИКАКИХ тяжёлых импортов (ни TensorFlow, ни самого FastAPI-app),
чтобы ``/healthz`` оставался дешёвым и отвечал даже когда ML-стек не поднят.
Версия берётся из модульной константы, а не из импорта пакета целиком.

Роут в ``faun/api.py`` подключает оркестратор отдельной волной:

    @app.get("/healthz")
    def healthz() -> dict:
        from faun.health import health
        return health()
"""

from __future__ import annotations

import os
from pathlib import Path

#: Версия сервиса для health-payload (release candidate v2).
FAUN_VERSION = "2.0.0-rc"

#: Имя сервиса в payload — совпадает с title FastAPI-app.
SERVICE_NAME = "faun-api"


def _jobs_root_ok() -> bool:
    """Дешёвая readiness-проверка: каталог job-ов существует и в него пишется.

    Повторяет резолв ``faun.api.jobs_root`` (env ``FAUN_JOBS_ROOT``, default
    ``./jobs``), но БЕЗ импорта api-модуля — чтобы handler не тянул FastAPI и
    оставался TF-free. Любое исключение глотаем и возвращаем ``False``: health
    обязан деградировать, а не падать.
    """
    try:
        root = Path(os.environ.get("FAUN_JOBS_ROOT", "./jobs"))
        root.mkdir(parents=True, exist_ok=True)
        return os.access(root, os.W_OK)
    except Exception:  # noqa: BLE001 — readiness НИКОГДА не должна бросать
        return False


def health() -> dict:
    """Вернуть liveness/readiness payload сервиса (без тяжёлых импортов).

    Всегда содержит ключи ``status``/``service``/``version``. Дополнительно —
    дешёвый readiness-флаг ``jobs_root_writable``: если каталог job-ов недоступен
    на запись, ``status`` деградирует до ``"degraded"`` (handler при этом НЕ
    бросает исключений — деградация вместо отказа).

    Returns:
        dict: ``{"status": "ok"|"degraded", "service": str, "version": str,
        "jobs_root_writable": bool}``.
    """
    jobs_ok = _jobs_root_ok()
    return {
        "status": "ok" if jobs_ok else "degraded",
        "service": SERVICE_NAME,
        "version": FAUN_VERSION,
        "jobs_root_writable": jobs_ok,
    }
