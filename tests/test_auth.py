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
# 5. Same-origin bypass (FAUN-37b/A): allowed Origin + Sec-Fetch-Site=same-origin
# ---------------------------------------------------------------------------


SAME_ORIGIN_HEADERS = {
    "Origin": "https://faun.antopkin.ru",
    "Sec-Fetch-Site": "same-origin",
}


@pytest.fixture
def env_with_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAUN_API_KEY + FAUN_FRONTEND_ORIGINS allowlisting the production origin."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://faun.antopkin.ru")


@pytest.mark.parametrize("method,path", PROTECTED_MUTATING_ROUTES, ids=_PARAM_IDS)
def test_origin_bypass_accepts_allowlisted_origin(
    method: str, path: str, env_with_origin: None
) -> None:
    """Allowlisted Origin + Sec-Fetch-Site=same-origin must bypass X-API-Key check."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers=SAME_ORIGIN_HEADERS)
    assert resp.status_code not in (401, 403, 503), (
        f"{method} {path}: same-origin bypass rejected with {resp.status_code}"
    )


@pytest.mark.parametrize("method,path", PROTECTED_MUTATING_ROUTES, ids=_PARAM_IDS)
def test_origin_bypass_rejects_wrong_origin(
    method: str, path: str, env_with_origin: None
) -> None:
    """Origin outside FAUN_FRONTEND_ORIGINS must NOT bypass auth — still 403."""
    headers = {"Origin": "https://evil.example", "Sec-Fetch-Site": "same-origin"}
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers=headers)
    assert resp.status_code == 403, (
        f"{method} {path}: wrong-origin bypass accepted with {resp.status_code}"
    )


@pytest.mark.parametrize("method,path", PROTECTED_MUTATING_ROUTES, ids=_PARAM_IDS)
def test_origin_bypass_requires_sec_fetch_site_same_origin(
    method: str, path: str, env_with_origin: None
) -> None:
    """Allowlisted Origin without Sec-Fetch-Site=same-origin must NOT bypass."""
    headers = {"Origin": "https://faun.antopkin.ru", "Sec-Fetch-Site": "cross-site"}
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers=headers)
    assert resp.status_code == 403, (
        f"{method} {path}: cross-site bypass accepted with {resp.status_code}"
    )


@pytest.mark.parametrize("method,path", PROTECTED_MUTATING_ROUTES, ids=_PARAM_IDS)
def test_origin_bypass_disabled_when_env_unset(
    method: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without FAUN_FRONTEND_ORIGINS, bypass must fail-closed even with same-origin headers."""
    monkeypatch.setenv("FAUN_API_KEY", "test_key_123")
    monkeypatch.delenv("FAUN_FRONTEND_ORIGINS", raising=False)
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers=SAME_ORIGIN_HEADERS)
    assert resp.status_code == 403, (
        f"{method} {path}: bypass accepted without FAUN_FRONTEND_ORIGINS env"
    )


@pytest.mark.parametrize("method,path", PROTECTED_MUTATING_ROUTES, ids=_PARAM_IDS)
def test_origin_bypass_rejects_origin_null(
    method: str, path: str, env_with_origin: None
) -> None:
    """Origin: null (sandboxed iframe, file://, data:) must NEVER bypass — attack vector."""
    headers = {"Origin": "null", "Sec-Fetch-Site": "same-origin"}
    client = TestClient(app, raise_server_exceptions=False)
    resp = _call(client, method, path, headers=headers)
    assert resp.status_code == 403, (
        f"{method} {path}: Origin: null bypass accepted with {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 6. Public GET routes stay open even when FAUN_API_KEY is set
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
