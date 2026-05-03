"""Auth middleware tests for FAUN-37 security hardening.

Tests verify that all mutating endpoints require X-API-Key matching FAUN_API_KEY env.
Public GET endpoints remain open.

CRITICAL: uses _get_real_app() to avoid mock injection from test_drone_bot.py /
test_bot_workflow.py which replace sys.modules["cloud.interface.main"] with MagicMock.
"""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient


def _get_real_app():
    """Get the real FastAPI app, even if other tests mocked the module."""
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811
    return sys.modules[mod_name].app


app = _get_real_app()


# 21 mutating endpoints that MUST require X-API-Key after FAUN-37 lands.
PROTECTED_MUTATING_ROUTES = [
    ("POST", "/api/v1/rangers"),
    ("DELETE", "/api/v1/rangers/{chat_id}"),
    ("PATCH", "/api/v1/rangers/{chat_id}/zone"),
    ("PATCH", "/api/v1/rangers/{chat_id}/active"),
    ("POST", "/api/v1/permits"),
    ("DELETE", "/api/v1/permits/{permit_id}"),
    ("POST", "/api/v1/permits/check"),
    ("POST", "/api/v1/rag-query"),
    ("POST", "/api/v1/classify"),
    ("POST", "/api/v1/agent/classify"),
    ("POST", "/api/v1/live/audio"),
    ("POST", "/api/v1/live/photo"),
    ("POST", "/api/v1/gateway-event"),
    ("POST", "/api/v1/mics/reseed"),
    ("PATCH", "/api/v1/mics/{mic_uid}/status"),
    ("PATCH", "/api/v1/mics/{mic_uid}/battery"),
    ("POST", "/api/v1/fgis-lk/violation"),
    ("POST", "/api/v1/workflow/run"),
    ("POST", "/api/v1/demo"),
    ("POST", "/demo/start"),
    ("POST", "/api/v1/incidents/{incident_id}/dispatch-drone"),
]


# Path-param substitutions: harmless test values.
PATH_PARAM_VALUES = {
    "chat_id": "123",
    "permit_id": "1",
    "mic_uid": "nonexistent_uid",
    "incident_id": "1",
}


def _resolve_path(path: str) -> str:
    """Substitute {param} placeholders with safe test values."""
    resolved = path
    for name, value in PATH_PARAM_VALUES.items():
        resolved = resolved.replace("{" + name + "}", value)
    return resolved


def _call(client: TestClient, method: str, path: str, headers: dict | None = None):
    """Issue a request with empty JSON body — auth check fires before validation."""
    url = _resolve_path(path)
    return client.request(method, url, json={}, headers=headers or {})


_PARAM_IDS = [f"{m} {p}" for m, p in PROTECTED_MUTATING_ROUTES]


# ---------------------------------------------------------------------------
# 1. FAUN_API_KEY env unset → expect 503 (not configured)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    PROTECTED_MUTATING_ROUTES,
    ids=_PARAM_IDS,
)
def test_mutating_route_requires_api_key_when_unset(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When FAUN_API_KEY is unset, mutating routes must respond 503."""
    monkeypatch.delenv("FAUN_API_KEY", raising=False)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path)
    assert resp.status_code == 503, (
        f"{method} {path}: expected 503 when FAUN_API_KEY unset, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 2. Missing X-API-Key header → expect 403
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    PROTECTED_MUTATING_ROUTES,
    ids=_PARAM_IDS,
)
def test_mutating_route_rejects_missing_api_key(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FAUN_API_KEY set and no X-API-Key header, route must respond 403."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path)
    assert resp.status_code == 403, (
        f"{method} {path}: expected 403 with no X-API-Key header, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 3. Wrong X-API-Key value → expect 403
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    PROTECTED_MUTATING_ROUTES,
    ids=_PARAM_IDS,
)
def test_mutating_route_rejects_wrong_api_key(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With X-API-Key not matching FAUN_API_KEY, route must respond 403."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers={"X-API-Key": "wrong_key"})
    assert resp.status_code == 403, (
        f"{method} {path}: expected 403 with wrong X-API-Key, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 4. Correct X-API-Key → auth passes (any non-401/403 status is fine)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    PROTECTED_MUTATING_ROUTES,
    ids=_PARAM_IDS,
)
def test_mutating_route_accepts_correct_api_key(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With matching X-API-Key, auth must not reject (status NOT in {401, 403})."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers={"X-API-Key": "test_key_123"})
    assert resp.status_code not in (401, 403), (
        f"{method} {path}: correct X-API-Key was rejected with {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 5. Public GET routes stay open even when FAUN_API_KEY is set
# ---------------------------------------------------------------------------


PUBLIC_GET_ROUTES = [
    "/health",
    "/",
    "/api/v1/mics",
    "/api/v1/incidents/export",
]


@pytest.mark.parametrize("path", PUBLIC_GET_ROUTES, ids=PUBLIC_GET_ROUTES)
def test_public_get_routes_remain_open(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public GET routes must respond without X-API-Key (status NOT in {401, 403})."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(path)
    assert resp.status_code not in (401, 403), (
        f"GET {path}: public route rejected with {resp.status_code}"
    )
