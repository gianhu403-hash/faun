"""Tests for the env-gated HTTP Basic Auth middleware (faun.api).

Default-OPEN: with no FAUN_BASIC_USER/PASS the site behaves exactly as before, so
the rest of the suite is unaffected. When BOTH are set, every path except
/healthz requires valid credentials (constant-time compared).
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from faun import api
from faun.settings import get_settings


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point jobs root at a temp dir and start from a clean auth env."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.delenv("FAUN_BASIC_USER", raising=False)
    monkeypatch.delenv("FAUN_BASIC_PASS", raising=False)
    get_settings.cache_clear()
    return monkeypatch


def _enable_auth(monkeypatch, user="ranger", pwd="forest42"):
    monkeypatch.setenv("FAUN_BASIC_USER", user)
    monkeypatch.setenv("FAUN_BASIC_PASS", pwd)
    get_settings.cache_clear()


def _basic(user, pwd):
    raw = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def test_auth_disabled_is_open(env):
    """No FAUN_BASIC_* -> open: / and POST /jobs are never 401."""
    client = TestClient(api.app)
    assert client.get("/").status_code != 401
    assert client.post("/jobs", json={"source_path": "/data/A1"}).status_code != 401
    assert client.get("/static/app.js").status_code != 401


def test_auth_enabled_rejects_without_creds(env):
    _enable_auth(env)
    client = TestClient(api.app)
    resp = client.get("/")
    assert resp.status_code == 401
    # Native browser login dialog hinges on this header.
    assert resp.headers.get("WWW-Authenticate", "").lower().startswith("basic")


def test_auth_enabled_accepts_correct_creds(env):
    _enable_auth(env, "ranger", "forest42")
    client = TestClient(api.app)
    assert client.get("/", headers=_basic("ranger", "forest42")).status_code == 200


def test_auth_enabled_rejects_wrong_creds(env):
    _enable_auth(env, "ranger", "forest42")
    client = TestClient(api.app)
    assert client.get("/", headers=_basic("ranger", "WRONG")).status_code == 401
    assert client.get("/", headers=_basic("intruder", "forest42")).status_code == 401


def test_auth_guards_static_mount(env):
    """The /static mount is behind auth too (a route dependency could not do this)."""
    _enable_auth(env)
    client = TestClient(api.app)
    assert client.get("/static/app.js").status_code == 401
    ok = client.get("/static/app.js", headers=_basic("ranger", "forest42"))
    assert ok.status_code == 200


def test_healthz_always_open_even_with_auth(env):
    """/healthz must stay open for the container HEALTHCHECK even when auth is on."""
    _enable_auth(env)
    client = TestClient(api.app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["service"] == "faun-api"


def test_whitespace_password_keeps_auth_enabled(env):
    """All-whitespace creds must NOT silently disable auth (fail CLOSED).

    Regression: reading creds via a path-style strip helper would turn "   " into
    None and flip the gate to open. The verbatim secret reader keeps it enabled.
    """
    env.setenv("FAUN_BASIC_USER", "ranger")
    env.setenv("FAUN_BASIC_PASS", "   ")  # whitespace, not empty
    get_settings.cache_clear()
    client = TestClient(api.app)
    assert client.get("/").status_code == 401  # auth still enforced
    assert client.get("/", headers=_basic("ranger", "   ")).status_code == 200


@pytest.mark.parametrize(
    "bad_header",
    [
        "Basic",  # no space / no payload
        "Basic !!!not-base64!!!",  # undecodable
        "Bearer abc",  # wrong scheme
        "Basic " + base64.b64encode(b"nocolon").decode(),  # no ':' separator
        "Basic " + base64.b64encode(b"ranger:wrong").decode(),  # wrong password
    ],
)
def test_auth_malformed_or_wrong_header_rejected(env, bad_header):
    """Garbage / wrong Authorization headers never bypass auth (401, no crash)."""
    _enable_auth(env, "ranger", "forest42")
    client = TestClient(api.app)
    assert client.get("/", headers={"Authorization": bad_header}).status_code == 401
