"""Rangers management API router."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cloud.db.rangers import (
    add_ranger,
    remove_ranger,
    get_all_rangers,
    update_zone,
    set_active,
)
from cloud.interface.security import require_api_key

router = APIRouter()


class RangerCreate(BaseModel):
    name: str
    chat_id: int
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float


class RangerZoneUpdate(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@router.get("/api/v1/rangers", dependencies=[Depends(require_api_key)])
async def list_rangers():
    """List all registered rangers."""
    rangers = get_all_rangers()
    return [
        {
            "id": r.id,
            "name": r.name,
            "chat_id": r.chat_id,
            "zone": {
                "lat_min": r.zone_lat_min,
                "lat_max": r.zone_lat_max,
                "lon_min": r.zone_lon_min,
                "lon_max": r.zone_lon_max,
            },
            "current_lat": r.current_lat,
            "current_lon": r.current_lon,
            "active": r.active,
        }
        for r in rangers
    ]


@router.post("/api/v1/rangers", dependencies=[Depends(require_api_key)])
async def create_ranger(req: RangerCreate):
    """Register a new ranger with their monitoring zone."""
    ranger = add_ranger(
        name=req.name,
        chat_id=req.chat_id,
        zone_lat_min=req.zone_lat_min,
        zone_lat_max=req.zone_lat_max,
        zone_lon_min=req.zone_lon_min,
        zone_lon_max=req.zone_lon_max,
    )
    return {"status": "created", "id": ranger.id, "name": ranger.name}


@router.delete("/api/v1/rangers/{chat_id}", dependencies=[Depends(require_api_key)])
async def delete_ranger(chat_id: int):
    """Remove a ranger by their Telegram chat ID."""
    removed = remove_ranger(chat_id)
    if not removed:
        return {"status": "not_found"}
    return {"status": "removed"}


@router.patch("/api/v1/rangers/{chat_id}/zone", dependencies=[Depends(require_api_key)])
async def update_ranger_zone(chat_id: int, req: RangerZoneUpdate):
    """Update a ranger's monitoring zone."""
    updated = update_zone(chat_id, req.lat_min, req.lat_max, req.lon_min, req.lon_max)
    if not updated:
        return {"status": "not_found"}
    return {"status": "updated"}


@router.patch(
    "/api/v1/rangers/{chat_id}/active", dependencies=[Depends(require_api_key)]
)
async def toggle_ranger_active(chat_id: int, active: bool = True):
    """Enable or disable alerts for a ranger."""
    updated = set_active(chat_id, active)
    if not updated:
        return {"status": "not_found"}
    return {"status": "active" if active else "inactive"}
