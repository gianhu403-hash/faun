"""Unit tests for cloud.interface.security helpers (no TestClient).

Covers _check_auth() reasons and is_authenticated() boolean variant directly,
including the empty/whitespace-only FAUN_API_KEY edge case that the parametrized
endpoint tests in test_auth.py do not exercise.
"""

from __future__ import annotations

import pytest

from cloud.interface.security import (
    _allowed_origins,
    _check_auth,
    _expected_key,
    _origin_allowed,
    _safe_for_log,
    is_authenticated,
)


# ---------------------------------------------------------------------------
# _expected_key — env reading + paste-artefact stripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain", "plain"),
        ("  spaced  ", "spaced"),
        ("trailing-newline\n", "trailing-newline"),
        ("\nleading-newline", "leading-newline"),
        ("inner space", "inner space"),
    ],
)
def test_expected_key_strips_paste_artefacts(
    raw: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAUN_API_KEY", raw)
    assert _expected_key() == expected


def test_expected_key_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAUN_API_KEY", raising=False)
    assert _expected_key() is None


@pytest.mark.parametrize("raw", ["", "   ", "\n", "\t  \n"])
def test_expected_key_empty_or_whitespace_returns_none(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAUN_API_KEY", raw)
    assert _expected_key() is None


# ---------------------------------------------------------------------------
# _check_auth — reason strings (table-driven)
# ---------------------------------------------------------------------------


def test_check_auth_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAUN_API_KEY", raising=False)
    assert _check_auth(None, None, None) == (False, "unconfigured")


def test_check_auth_configured_but_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "")
    assert _check_auth("anything", None, None) == (False, "configured_but_empty")


def test_check_auth_configured_but_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "   \n")
    assert _check_auth("anything", None, None) == (False, "configured_but_empty")


def test_check_auth_missing_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    assert _check_auth(None, None, None) == (False, "missing_header")


def test_check_auth_key_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    assert _check_auth("wrong", None, None) == (False, "key_mismatch")


def test_check_auth_key_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    assert _check_auth("real", None, None) == (True, "key_match")


def test_check_auth_key_match_strips_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    assert _check_auth("real\n", None, None) == (True, "key_match")


def test_check_auth_origin_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://faun.example")
    assert _check_auth(None, "https://faun.example", "same-origin") == (
        True,
        "origin_bypass",
    )


# ---------------------------------------------------------------------------
# _origin_allowed — corner cases
# ---------------------------------------------------------------------------


def test_origin_allowed_multi_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://a.example, https://b.example")
    assert _origin_allowed("https://a.example", "same-origin") is True
    assert _origin_allowed("https://b.example", "same-origin") is True
    assert _origin_allowed("https://c.example", "same-origin") is False


def test_origin_allowed_rejects_null_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if admin allowlists 'null' explicitly, sandboxed iframe must not bypass."""
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "null")
    assert _origin_allowed("null", "same-origin") is False


def test_origin_allowed_case_sensitive_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browsers send lowercase host per RFC 6454; uppercase is admin typo, fail-closed."""
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://faun.example")
    assert _origin_allowed("https://FAUN.example", "same-origin") is False


def test_origin_allowed_rejects_cross_site(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://faun.example")
    assert _origin_allowed("https://faun.example", "cross-site") is False
    assert _origin_allowed("https://faun.example", "same-site") is False
    assert _origin_allowed("https://faun.example", None) is False


def test_allowed_origins_parses_csv_with_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", " a , b ,, c, ")
    assert _allowed_origins() == {"a", "b", "c"}


def test_allowed_origins_unset_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAUN_FRONTEND_ORIGINS", raising=False)
    assert _allowed_origins() == set()


# ---------------------------------------------------------------------------
# is_authenticated — boolean variant matches _check_auth
# ---------------------------------------------------------------------------


def test_is_authenticated_true_when_key_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    assert is_authenticated("real", None, None) is True


def test_is_authenticated_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAUN_API_KEY", raising=False)
    assert is_authenticated("real", None, None) is False


def test_is_authenticated_true_via_origin_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAUN_API_KEY", "real")
    monkeypatch.setenv("FAUN_FRONTEND_ORIGINS", "https://faun.example")
    assert is_authenticated(None, "https://faun.example", "same-origin") is True


# ---------------------------------------------------------------------------
# _safe_for_log — log-injection sanitization
# ---------------------------------------------------------------------------


def test_safe_for_log_strips_crlf() -> None:
    assert _safe_for_log("foo\nbar") == "foo\\nbar"
    assert _safe_for_log("foo\r\nbar") == "foo\\r\\nbar"


def test_safe_for_log_truncates() -> None:
    assert _safe_for_log("a" * 500, limit=10) == "a" * 10


def test_safe_for_log_none_returns_dash() -> None:
    assert _safe_for_log(None) == "-"


def test_safe_for_log_coerces_non_str() -> None:
    """Defensive: accept bytes / int / unexpected types without TypeError."""
    assert _safe_for_log(b"bytes") == "b'bytes'"
    assert _safe_for_log(42) == "42"


# ---------------------------------------------------------------------------
# Audit-log contract: each rejection reason must hit the logger
# ---------------------------------------------------------------------------


import logging  # noqa: E402

from fastapi import FastAPI, Depends  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from cloud.interface.security import require_api_key  # noqa: E402


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.post("/protected", dependencies=[Depends(require_api_key)])
    def _ep() -> dict[str, str]:
        return {"ok": "yes"}

    return app


@pytest.mark.parametrize(
    "headers,env,expected_reason,expected_status",
    [
        ({}, {}, "unconfigured", 503),
        ({}, {"FAUN_API_KEY": ""}, "configured_but_empty", 503),
        ({}, {"FAUN_API_KEY": "real"}, "missing_header", 403),
        ({"X-API-Key": "wrong"}, {"FAUN_API_KEY": "real"}, "key_mismatch", 403),
    ],
)
def test_rejection_emits_audit_log(
    headers: dict[str, str],
    env: dict[str, str],
    expected_reason: str,
    expected_status: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each rejection path must write a log line carrying the reason — audit trail contract."""
    monkeypatch.delenv("FAUN_API_KEY", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    client = TestClient(_make_app(), raise_server_exceptions=False)
    with caplog.at_level(logging.WARNING, logger="cloud.interface.security"):
        resp = client.post("/protected", headers=headers)
    assert resp.status_code == expected_status
    assert any(expected_reason in record.getMessage() for record in caplog.records), (
        f"expected '{expected_reason}' in logs, got: {[r.getMessage() for r in caplog.records]}"
    )
