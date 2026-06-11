"""Shared state and utilities for bot handlers.

This module is the single source of truth for:
- ADMIN_CHAT_IDS (parsed from env)
- _registration_state (mutable dict, shared across handlers)
- Registration step constants and TTL
- _safe_answer() and _haversine() utilities
"""

import math
import os

# ---------- Admin chat IDs ----------

ADMIN_CHAT_IDS: set[int] = set()
_admin_env = os.getenv("ADMIN_CHAT_IDS", "")
if _admin_env:
    ADMIN_CHAT_IDS = {int(x.strip()) for x in _admin_env.split(",") if x.strip()}


# ---------- Registration state (manual, no ConversationHandler) ----------

_registration_state: dict[int, dict] = {}
_REG_STEP_NAME = "awaiting_name"
_REG_STEP_BADGE = "awaiting_badge"
_REG_STEP_CONFIRM = "awaiting_confirm"
_REG_TTL = 1800  # 30 minutes


# ---------- Utilities ----------


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _safe_answer(query) -> None:
    """answer() just removes the loading spinner — non-critical."""
    try:
        await query.answer()
    except Exception:
        pass
