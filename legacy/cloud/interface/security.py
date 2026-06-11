"""API key authentication with optional same-origin frontend bypass.

FAUN-37    — write-side enforcement (require_api_key on 21 mutation endpoints).
FAUN-37b/A — same-origin bypass via Origin + Sec-Fetch-Site=same-origin so the
             browser-based dashboard can POST without embedding the API key in
             HTML (which would be an XSS-leak surface).
             Also exposes is_authenticated() for read-side conditional redaction.

Accepted header spellings (FastAPI/HTTP normalize case):
- X-API-Key / x-api-key   → x_api_key
- Origin                  → origin
- Sec-Fetch-Site          → sec_fetch_site (alias)

Threat model: bypass closes "random scanner POSTs to /api/v1/demo from outside".
Spoofing Origin+Sec-Fetch-Site requires a non-browser client (curl, requests) —
inside browsers both are forbidden header names that JS (incl. XSS payloads)
cannot override. So the bypass is safer than embedding the key in HTML.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


def _allowed_origins() -> set[str]:
    raw = os.getenv("FAUN_FRONTEND_ORIGINS", "")
    return {o.strip() for o in raw.split(",") if o.strip()}


def _expected_key() -> str | None:
    """Return FAUN_API_KEY stripped of paste artefacts (newline, surrounding ws).

    Operators routinely paste keys with trailing \\n from `pbcopy` / web UIs;
    silently treating " key " as != "key" creates support burden.
    Returns None if env unset OR effectively empty after strip.
    """
    raw = os.getenv("FAUN_API_KEY")
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _origin_allowed(origin: str | None, sec_fetch_site: str | None) -> bool:
    """Same-origin bypass eligibility.

    Requires: non-empty, non-"null" Origin in FAUN_FRONTEND_ORIGINS allowlist
              AND Sec-Fetch-Site=same-origin (browsers always send both for fetch).
    Empty/missing FAUN_FRONTEND_ORIGINS → bypass disabled (fail-closed).
    "Origin: null" (sandboxed iframe, file://, data:) is always rejected even
    if the admin accidentally allowlisted "null" — known CSRF attack vector.
    """
    if not origin or origin == "null" or sec_fetch_site != "same-origin":
        return False
    allowed = _allowed_origins()
    if not allowed:
        return False
    return origin in allowed


def _check_auth(
    x_api_key: str | None,
    origin: str | None,
    sec_fetch_site: str | None,
) -> tuple[bool, str]:
    """Core auth shared by require_api_key (raises) and is_authenticated (returns).

    Checks BOTH X-API-Key AND same-origin bypass — name reflects that this is
    the auth gate, not just key validation.

    Returns (ok, reason). reason is for server-side logging only — never client-exposed.
    """
    expected = _expected_key()
    if expected is None:
        if "FAUN_API_KEY" in os.environ:
            return False, "configured_but_empty"
        return False, "unconfigured"
    if _origin_allowed(origin, sec_fetch_site):
        return True, "origin_bypass"
    if not x_api_key:
        return False, "missing_header"
    candidate = x_api_key.strip()
    if not secrets.compare_digest(candidate, expected):
        return False, "key_mismatch"
    return True, "key_match"


def is_authenticated(
    x_api_key: str | None,
    origin: str | None,
    sec_fetch_site: str | None,
) -> bool:
    """Boolean variant — never raises.

    Use for conditional payload redaction (e.g. mics endpoint shows full data
    to authed callers, public-safe subset to anyone else).
    """
    ok, _ = _check_auth(x_api_key, origin, sec_fetch_site)
    return ok


def _safe_for_log(value: object, limit: int = 200) -> str:
    """Strip CR/LF and cap length — prevents log forging via Origin header injection.

    Accepts any type (defensive against future Header subclasses returning bytes
    or custom shims passing None for request.client.host); coerces to str first.
    """
    if value is None:
        return "-"
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\r", "\\r").replace("\n", "\\n")[:limit]


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    origin: str | None = Header(default=None),
    sec_fetch_site: str | None = Header(default=None, alias="Sec-Fetch-Site"),
) -> None:
    """FastAPI dependency: enforce X-API-Key OR same-origin bypass.

    Raises 503 if FAUN_API_KEY is unset or empty (fail-closed; generic detail).
    Raises 403 if neither valid X-API-Key nor allowed Origin+Sec-Fetch-Site.
    Logs all rejections with reason+path+remote for audit trail.
    """
    ok, reason = _check_auth(x_api_key, origin, sec_fetch_site)
    if ok:
        return
    client_ip = _safe_for_log(request.client.host) if request.client else "-"
    path = _safe_for_log(request.url.path)
    if reason in ("unconfigured", "configured_but_empty"):
        logger.error(
            "auth blocked: FAUN_API_KEY %s (path=%s remote=%s)",
            reason,
            path,
            client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service unavailable",
        )
    logger.warning(
        "auth rejected: %s (path=%s remote=%s origin=%s)",
        reason,
        path,
        client_ip,
        _safe_for_log(origin),
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="forbidden",
    )
