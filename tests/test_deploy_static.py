"""Static lint for the deploy/ artifacts (Docker НЕ запускается).

Это статическая проверка: читаем deploy/Dockerfile и валидируем обязательные
директивы — тест РЕАЛЬНО падает на битом Dockerfile. docker-compose.yml парсим
как YAML опционально (pyyaml может быть не установлен → importorskip).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
_DOCKERFILE = _DEPLOY / "Dockerfile"
_COMPOSE = _DEPLOY / "docker-compose.yml"


def _dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    return _DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_starts_with_from() -> None:
    """Первая значимая инструкция Dockerfile должна быть FROM."""
    text = _dockerfile_text()
    # Пропускаем пустые строки и комментарии до первой инструкции.
    instructions = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert instructions, "Dockerfile has no instructions"
    assert instructions[0].upper().startswith("FROM "), (
        f"Dockerfile must start with FROM, got: {instructions[0]!r}"
    )


def test_dockerfile_has_required_directives() -> None:
    """Обязательные директивы: FROM, COPY/ADD, EXPOSE, CMD/ENTRYPOINT."""
    text = _dockerfile_text()
    upper = text.upper()
    assert "FROM " in upper, "missing FROM"
    assert ("COPY " in upper) or ("ADD " in upper), "missing COPY/ADD"
    assert "EXPOSE " in upper, "missing EXPOSE"
    assert ("CMD " in upper) or ("CMD[" in upper) or ("ENTRYPOINT" in upper), (
        "missing CMD/ENTRYPOINT"
    )


def test_dockerfile_installs_pipeline_requirements() -> None:
    """Образ ставит именно requirements-pipeline.txt (лёгкий, без TF)."""
    text = _dockerfile_text()
    assert "requirements-pipeline.txt" in text, (
        "Dockerfile must install requirements-pipeline.txt"
    )


def test_dockerfile_runs_uvicorn_app() -> None:
    """Контейнер запускает uvicorn faun.api:app."""
    text = _dockerfile_text()
    assert "uvicorn" in text, "Dockerfile must run uvicorn"
    assert "faun.api:app" in text, "Dockerfile must serve faun.api:app"


def test_dockerfile_has_healthcheck_on_healthz() -> None:
    """HEALTHCHECK должен дёргать /healthz."""
    text = _dockerfile_text()
    assert "HEALTHCHECK" in text.upper(), "missing HEALTHCHECK"
    assert "/healthz" in text, "HEALTHCHECK must hit /healthz"


def test_dockerfile_avoids_v1_demo_ports() -> None:
    """v2 не должен EXPOSE-ить порты демо v1 (8003/8005/9000)."""
    text = _dockerfile_text()
    for collide in ("8003", "8005", "9000"):
        assert f"EXPOSE {collide}" not in text.upper(), (
            f"port {collide} занят демо v1, не переиспользовать"
        )


def test_compose_parses_as_yaml() -> None:
    """docker-compose.yml — валидный YAML с сервисом faun-api на 8010."""
    yaml = pytest.importorskip("yaml")
    assert _COMPOSE.is_file(), f"missing {_COMPOSE}"
    data = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    assert "services" in data, "compose has no services"
    assert "faun-api" in data["services"], "compose missing faun-api service"
    ports = data["services"]["faun-api"].get("ports", [])
    assert any("8010" in str(p) for p in ports), "faun-api must map port 8010"
