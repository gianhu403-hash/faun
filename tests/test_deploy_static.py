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
_DOCKERFILE_ML = _DEPLOY / "Dockerfile.ml"
_COMPOSE = _DEPLOY / "docker-compose.yml"
_ROOT = _DEPLOY.parent


def _dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    return _DOCKERFILE.read_text(encoding="utf-8")


def _dockerfile_ml_text() -> str:
    assert _DOCKERFILE_ML.is_file(), f"missing {_DOCKERFILE_ML}"
    return _DOCKERFILE_ML.read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Dockerfile.ml — TF + real Perch 2 serving image (additive; slim stays the rollback)
# ---------------------------------------------------------------------------


def test_dockerfile_ml_exists_and_starts_with_from() -> None:
    """Dockerfile.ml exists and its first instruction is FROM."""
    text = _dockerfile_ml_text()
    instructions = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert instructions, "Dockerfile.ml has no instructions"
    assert instructions[0].upper().startswith("FROM "), (
        f"Dockerfile.ml must start with FROM, got: {instructions[0]!r}"
    )


def test_dockerfile_ml_installs_both_requirements() -> None:
    """The ML image layers requirements-ml.txt ON TOP of the pipeline stack."""
    text = _dockerfile_ml_text()
    assert "requirements-pipeline.txt" in text, "must install the base pipeline stack"
    assert "requirements-ml.txt" in text, "must install requirements-ml.txt (TF layer)"


def test_dockerfile_ml_defaults_to_real_perch_v2() -> None:
    """The ML image defaults to the real Perch 2 classifier + model path."""
    text = _dockerfile_ml_text()
    assert "FAUN_CLASSIFIER=perch-v2" in text, "ML image must default to perch-v2"
    assert "PERCH_V2_MODEL_PATH=/models/perch2" in text, (
        "must point at the model volume"
    )


def test_dockerfile_ml_copies_experiments_wrappers() -> None:
    """Perch2Adapter._infer delegates to experiments.wrappers.perch_v2 — it must ship."""
    text = _dockerfile_ml_text()
    assert "experiments/wrappers" in text, (
        "Dockerfile.ml must COPY experiments/wrappers (the Perch 2 _infer delegate)"
    )


def test_dockerfile_ml_serves_uvicorn_8010_with_healthz() -> None:
    """Same serving contract as slim: uvicorn faun.api:app, EXPOSE 8010, /healthz."""
    text = _dockerfile_ml_text()
    upper = text.upper()
    assert "uvicorn" in text and "faun.api:app" in text, "must serve faun.api:app"
    assert "EXPOSE 8010" in upper, "must EXPOSE 8010"
    assert "HEALTHCHECK" in upper and "/healthz" in text, "must HEALTHCHECK /healthz"
    for collide in ("8003", "8005", "9000"):
        assert f"EXPOSE {collide}" not in upper, f"port {collide} занят демо v1"


def test_requirements_ml_pins_tensorflow_and_kagglehub() -> None:
    """requirements-ml.txt pins a TF >= 2.20 build + kagglehub (Perch 2 needs both)."""
    req = (_ROOT / "requirements-ml.txt").read_text(encoding="utf-8")
    assert "tensorflow" in req, "requirements-ml.txt must pin tensorflow"
    assert "kagglehub" in req, "requirements-ml.txt must pin kagglehub"
    # The TF floor that Perch 2 requires.
    assert "2.2" in req, "TF must be >= 2.20 for Perch 2"


def test_slim_dockerfile_stays_tf_free() -> None:
    """ROLLBACK INTEGRITY: the slim image must NOT pull TF / the ML layer.

    The slim deploy/Dockerfile is the instant rollback (Stub classifier). If it
    ever started installing tensorflow or requirements-ml.txt it would stop being
    a lightweight, fast, TF-free fallback. We inspect only NON-comment instruction
    lines (the header comment legitimately says "без tensorflow").
    """
    text = _dockerfile_text()
    instructions = "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("#")
    ).lower()
    assert "tensorflow" not in instructions, "slim image must stay TensorFlow-free"
    assert "requirements-ml.txt" not in instructions, (
        "slim image must not install the ML layer"
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
