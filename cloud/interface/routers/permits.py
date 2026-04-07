"""Permits (лесные билеты) API router."""

from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel

from cloud.db.permits import (
    add_permit as db_add_permit,
    remove_permit as db_remove_permit,
    get_all_permits,
    get_permits_for_location,
    has_valid_permit,
)

router = APIRouter()


class PermitCreate(BaseModel):
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float
    valid_from: date
    valid_until: date
    description: str = ""


class PermitCheck(BaseModel):
    lat: float
    lon: float


@router.get("/api/v1/permits")
async def list_permits():
    """List all logging permits."""
    permits = get_all_permits()
    return [
        {
            "id": p.id,
            "description": p.description,
            "zone": {
                "lat_min": p.zone_lat_min,
                "lat_max": p.zone_lat_max,
                "lon_min": p.zone_lon_min,
                "lon_max": p.zone_lon_max,
            },
            "valid_from": p.valid_from.isoformat(),
            "valid_until": p.valid_until.isoformat(),
        }
        for p in permits
    ]


@router.post("/api/v1/permits")
async def create_permit(req: PermitCreate):
    """Register a new logging permit."""
    permit = db_add_permit(
        zone_lat_min=req.zone_lat_min,
        zone_lat_max=req.zone_lat_max,
        zone_lon_min=req.zone_lon_min,
        zone_lon_max=req.zone_lon_max,
        valid_from=req.valid_from,
        valid_until=req.valid_until,
        description=req.description,
    )
    return {"status": "created", "id": permit.id}


@router.delete("/api/v1/permits/{permit_id}")
async def delete_permit(permit_id: int):
    """Remove a logging permit."""
    removed = db_remove_permit(permit_id)
    if not removed:
        return {"status": "not_found"}
    return {"status": "removed"}


@router.post("/api/v1/permits/check")
async def check_permit(req: PermitCheck):
    """Check if a location is covered by a valid permit."""
    has_permit = has_valid_permit(req.lat, req.lon)
    permits = get_permits_for_location(req.lat, req.lon)
    return {
        "has_valid_permit": has_permit,
        "permits": [{"id": p.id, "description": p.description} for p in permits],
    }
