"""Microphone network API router."""

import concurrent.futures
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cloud.db.microphones import (
    get_all as mic_get_all,
    get_online as mic_get_online,
    set_status as mic_set_status,
    set_battery as mic_set_battery,
    clear_all as mic_clear_all,
    seed_microphones,
)
from fastapi import Header

from cloud.interface.security import is_authenticated, require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

_reseed_status: dict = {"running": False, "deleted": 0, "created": 0}


class MicStatusUpdate(BaseModel):
    status: str  # online, offline, broken


class MicBatteryUpdate(BaseModel):
    battery_pct: float


def _mic_dict(m, authed: bool) -> dict:
    """Public-safe by default; full payload only for authed callers.

    Public fields needed by Leaflet map: position + zone_type + online dot color.
    Authed-only fields are operational telemetry that doubles as a poacher's
    "where won't they hear me" map: battery_pct, granular status, install date,
    sub-district name (FAUN-37b/B).
    """
    base = {
        "mic_uid": m.mic_uid,
        "lat": m.lat,
        "lon": m.lon,
        "zone_type": m.zone_type,
        "online": m.status == "online",
    }
    if authed:
        base.update(
            {
                "status": m.status,
                "battery_pct": m.battery_pct,
                "installed_at": m.installed_at,
                "sub_district": m.sub_district,
            }
        )
    return base


@router.get("/api/v1/mics")
async def list_mics(
    x_api_key: str | None = Header(default=None),
    origin: str | None = Header(default=None),
    sec_fetch_site: str | None = Header(default=None, alias="Sec-Fetch-Site"),
):
    """List all microphones in the network. Telemetry redacted for unauth callers."""
    authed = is_authenticated(x_api_key, origin, sec_fetch_site)
    return [_mic_dict(m, authed) for m in mic_get_all()]


@router.get("/api/v1/mics/online", dependencies=[Depends(require_api_key)])
async def list_mics_online():
    """List only online microphones.

    Authed-only: combined with public /api/v1/mics this would let a poacher
    diff online vs all to get the exact "where can they hear me" map even
    after FAUN-37b/B redaction.
    """
    mics = mic_get_online()
    return [
        {"mic_uid": m.mic_uid, "lat": m.lat, "lon": m.lon, "zone_type": m.zone_type}
        for m in mics
    ]


@router.post("/api/v1/mics/reseed", dependencies=[Depends(require_api_key)])
async def reseed_mics():
    """Delete all microphones and re-seed from updated grid parameters.

    Runs seeding in a background thread to avoid nginx timeout.
    Poll GET /api/v1/mics/reseed/status for progress.
    """
    if _reseed_status["running"]:
        return {"status": "already_running"}

    def _do_reseed():
        _reseed_status["running"] = True
        try:
            deleted = mic_clear_all()
            _reseed_status["deleted"] = deleted
            logger.info("Cleared %d microphones, re-seeding...", deleted)
            mics = seed_microphones()
            _reseed_status["created"] = len(mics)
            logger.info("Re-seeded %d microphones", len(mics))
        finally:
            _reseed_status["running"] = False

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_do_reseed)
    return {"status": "started", "message": "Reseed running in background"}


@router.get("/api/v1/mics/reseed/status", dependencies=[Depends(require_api_key)])
async def reseed_status():
    """Check reseed progress (op telemetry — leaks operator activity windows)."""
    return _reseed_status


@router.patch("/api/v1/mics/{mic_uid}/status", dependencies=[Depends(require_api_key)])
async def update_mic_status(mic_uid: str, req: MicStatusUpdate):
    """Update microphone status."""
    ok = mic_set_status(mic_uid, req.status)
    if not ok:
        return {"status": "not_found"}
    return {"status": "updated", "mic_uid": mic_uid, "new_status": req.status}


@router.patch("/api/v1/mics/{mic_uid}/battery", dependencies=[Depends(require_api_key)])
async def update_mic_battery(mic_uid: str, req: MicBatteryUpdate):
    """Update microphone battery percentage."""
    ok = mic_set_battery(mic_uid, req.battery_pct)
    if not ok:
        return {"status": "not_found"}
    return {"status": "updated", "mic_uid": mic_uid, "battery_pct": req.battery_pct}
