"""FGIS-LK integration API router (stub)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cloud.integrations.fgis_lk import fgis_client, ViolationReport
from cloud.interface.security import require_api_key

router = APIRouter()


class ViolationSubmit(BaseModel):
    incident_id: str
    audio_class: str
    lat: float
    lon: float
    confidence: float
    ranger_name: str = ""
    description: str = ""


@router.get("/api/v1/fgis-lk/forest-unit")
async def fgis_forest_unit(lat: float, lon: float):
    """Look up forest quarter by coordinates (FGIS-LK stub)."""
    unit = fgis_client.get_forest_unit(lat, lon)
    return {
        "quarter_number": unit.quarter_number,
        "sub_district": unit.sub_district,
        "species_composition": unit.species_composition,
        "zone_type": unit.zone_type,
        "area_ha": unit.area_ha,
    }


@router.get("/api/v1/fgis-lk/permits", dependencies=[Depends(require_api_key)])
async def fgis_permits(lat: float, lon: float):
    """Get active felling permits for location (FGIS-LK stub)."""
    permits = fgis_client.get_active_permits(lat, lon)
    return [
        {
            "permit_id": p.permit_id,
            "felling_type": p.felling_type,
            "volume_m3": p.volume_m3,
            "contractor": p.contractor,
            "valid_from": p.valid_from.isoformat(),
            "valid_until": p.valid_until.isoformat(),
        }
        for p in permits
    ]


@router.post("/api/v1/fgis-lk/violation", dependencies=[Depends(require_api_key)])
async def fgis_violation(req: ViolationSubmit):
    """Submit violation report to FGIS-LK (stub)."""
    report = ViolationReport(
        incident_id=req.incident_id,
        audio_class=req.audio_class,
        lat=req.lat,
        lon=req.lon,
        confidence=req.confidence,
        ranger_name=req.ranger_name,
        description=req.description,
        timestamp="",
    )
    result = fgis_client.submit_violation(report)
    return result
