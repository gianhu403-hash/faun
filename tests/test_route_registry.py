"""Smoke tests: verify all routes in main.py are registered.

Safety net for the main.py → routers split refactoring.
Checks route registration, NOT business logic.
"""

import pytest

from cloud.interface.main import app


# Every route in main.py as of pre-split baseline
EXPECTED_ROUTES = [
    # Core
    ("GET", "/health"),
    ("GET", "/"),
    ("GET", "/analytics"),
    # Incidents
    ("GET", "/api/v1/incidents/{incident_id}/protocol.pdf"),
    ("POST", "/api/v1/incidents/{incident_id}/dispatch-drone"),
    # Demo
    ("POST", "/api/v1/demo"),
    ("POST", "/demo/start"),
    # Rangers
    ("GET", "/api/v1/rangers"),
    ("POST", "/api/v1/rangers"),
    ("DELETE", "/api/v1/rangers/{chat_id}"),
    ("PATCH", "/api/v1/rangers/{chat_id}/zone"),
    ("PATCH", "/api/v1/rangers/{chat_id}/active"),
    # Permits
    ("GET", "/api/v1/permits"),
    ("POST", "/api/v1/permits"),
    ("DELETE", "/api/v1/permits/{permit_id}"),
    ("POST", "/api/v1/permits/check"),
    # RAG
    ("POST", "/api/v1/rag-query"),
    # Classification
    ("POST", "/api/v1/classify"),
    ("POST", "/api/v1/agent/classify"),
    # Live
    ("POST", "/api/v1/live/audio"),
    ("POST", "/api/v1/live/photo"),
    # DataLens / Analytics
    ("GET", "/api/v1/incidents/export"),
    ("GET", "/api/v1/ai-studio-stack"),
    ("GET", "/api/v1/datalens/incidents"),
    # Gateway
    ("POST", "/api/v1/gateway-event"),
    # Mics
    ("GET", "/api/v1/mics"),
    ("GET", "/api/v1/mics/online"),
    ("POST", "/api/v1/mics/reseed"),
    ("GET", "/api/v1/mics/reseed/status"),
    ("PATCH", "/api/v1/mics/{mic_uid}/status"),
    ("PATCH", "/api/v1/mics/{mic_uid}/battery"),
    # FGIS-LK
    ("GET", "/api/v1/fgis-lk/forest-unit"),
    ("GET", "/api/v1/fgis-lk/permits"),
    ("POST", "/api/v1/fgis-lk/violation"),
    # Workflows
    ("GET", "/api/v1/workflow/definition"),
    ("POST", "/api/v1/workflow/run"),
]


def _get_registered_routes(fastapi_app):
    """Extract (method, path) pairs from FastAPI app."""
    routes = []
    for route in fastapi_app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            for method in route.methods:
                if method in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                    routes.append((method, route.path))
    return routes


class TestRouteRegistry:
    """Verify all expected routes are registered — safety net for split."""

    @pytest.mark.parametrize(
        "method,path",
        EXPECTED_ROUTES,
        ids=[f"{m} {p}" for m, p in EXPECTED_ROUTES],
    )
    def test_route_registered(self, method, path):
        registered = _get_registered_routes(app)
        assert (method, path) in registered, (
            f"Route {method} {path} not registered in app"
        )

    def test_total_route_count(self):
        """Catch accidental route loss — total must match expected."""
        registered = _get_registered_routes(app)
        assert len(registered) >= len(EXPECTED_ROUTES), (
            f"Expected at least {len(EXPECTED_ROUTES)} routes, got {len(registered)}"
        )

    def test_websocket_route_exists(self):
        """WebSocket /ws must be registered."""
        ws_paths = [
            r.path
            for r in app.routes
            if hasattr(r, "path") and not hasattr(r, "methods") and r.path == "/ws"
        ]
        assert "/ws" in ws_paths, "WebSocket /ws not registered"


class TestHTTPSmoke:
    """HTTP-level smoke tests for endpoints with no external deps."""

    def test_health_returns_ok(self):
        from starlette.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ai_studio_stack_returns_list(self):
        from starlette.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/ai-studio-stack")
        assert resp.status_code == 200
        data = resp.json()
        assert "integrations" in data
        assert len(data["integrations"]) == 9

    def test_reseed_status_returns_dict(self):
        from starlette.testclient import TestClient

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/mics/reseed/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
