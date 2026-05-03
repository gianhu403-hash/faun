"""API key authentication for FAUN-37."""

import os
import secrets
from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce X-API-Key matches FAUN_API_KEY env.

    Raises 503 if FAUN_API_KEY is not configured (fail closed).
    Raises 403 if header is missing or wrong (constant-time compare).
    """
    expected = os.getenv("FAUN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FAUN_API_KEY not configured",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid api key",
        )
