"""Classification and live inference API router."""

import asyncio
import logging
import os
import uuid

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from cloud.agent.classification_agent import verify_classification
from cloud.agent.datasphere_client import classify_embeddings

router = APIRouter()
logger = logging.getLogger(__name__)


class ClassifyRequest(BaseModel):
    embeddings: list[float]


class ClassifyAgentRequest(BaseModel):
    audio_class: str
    confidence: float
    lat: float
    lon: float
    zone_type: str = "exploitation"
    ndsi: float | None = None


@router.post("/api/v1/classify")
async def classify_cloud(req: ClassifyRequest):
    """Cloud classification via DataSphere Node (2-tier verification)."""
    result = await classify_embeddings(req.embeddings)
    if result is None:
        return {"status": "unavailable", "message": "DataSphere Node not configured"}
    return {"status": "ok", **result}


@router.post("/api/v1/agent/classify")
async def classify_agent(req: ClassifyAgentRequest):
    """Real-time classification verification via AI Studio agent."""
    result = await verify_classification(
        audio_class=req.audio_class,
        confidence=req.confidence,
        lat=req.lat,
        lon=req.lon,
        zone_type=req.zone_type,
        ndsi=req.ndsi,
    )
    return {
        "verified_class": result.verified_class,
        "confidence": result.confidence,
        "priority": result.priority,
        "context_analysis": result.context_analysis,
        "recommended_action": result.recommended_action,
        "permit_status": result.permit_status,
    }


def _get_broadcast():
    """Lazy import to avoid circular dependency."""
    from cloud.interface.main import broadcast

    return broadcast


def _get_classify_via_edge():
    """Lazy import to avoid circular dependency."""
    from cloud.interface.main import _classify_via_edge

    return _classify_via_edge


@router.post("/api/v1/live/audio")
async def live_audio(file: UploadFile):
    """Classify audio chunk from browser mic. Converts webm->wav via ffmpeg."""
    import subprocess

    broadcast = _get_broadcast()
    classify_via_edge = _get_classify_via_edge()

    webm_path = f"/tmp/live_{uuid.uuid4()}.webm"
    wav_path = f"/tmp/live_{uuid.uuid4()}.wav"
    content = await file.read()
    with open(webm_path, "wb") as f:
        f.write(content)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                webm_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-af",
                "highpass=f=100",
                wav_path,
            ],
            capture_output=True,
            timeout=10,
        )
        from cloud.notify.telegram import send_pending

        result = classify_via_edge(wav_path)
        await broadcast(
            {
                "event": "audio_classified",
                "class": result.label,
                "confidence": result.confidence,
            }
        )
        # Trigger alert pipeline if threat detected
        if result.label not in ("background", "unknown") and result.confidence >= 0.5:
            await send_pending(
                lat=57.3700,
                lon=44.6300,
                audio_class=result.label,
                reason=f"Live mic: {result.label}",
                confidence=result.confidence,
                is_demo=True,
            )
        return {"audio_class": result.label, "confidence": result.confidence}
    finally:
        for p in (webm_path, wav_path):
            if os.path.exists(p):
                os.unlink(p)


@router.post("/api/v1/live/photo")
async def live_photo(file: UploadFile):
    """Classify photo from browser camera via Gemma/YandexGPT Vision."""
    import base64

    from cloud.vision.classifier import classify_photo

    broadcast = _get_broadcast()

    content = await file.read()
    photo_b64 = base64.b64encode(content).decode()
    result = await classify_photo(photo_b64)
    await broadcast(
        {
            "event": "vision_classified",
            "description": result.description,
            "has_human": result.has_human,
            "has_fire": result.has_fire,
            "has_felling": result.has_felling,
            "has_machinery": result.has_machinery,
            "is_threat": result.is_threat,
            "time_of_day": result.time_of_day,
            "people_count": result.people_count,
            "equipment_types": result.equipment_types,
            "vegetation_damage": result.vegetation_damage,
            "damage_area_estimate": result.damage_area_estimate,
        }
    )
    return {
        "description": result.description,
        "has_human": result.has_human,
        "has_fire": result.has_fire,
        "has_felling": result.has_felling,
        "is_threat": result.is_threat,
    }
