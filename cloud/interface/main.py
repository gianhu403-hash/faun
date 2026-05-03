import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from cloud.interface.security import require_api_key
from cloud.interface.routers.rangers import router as rangers_router
from cloud.interface.routers.permits import router as permits_router
from cloud.interface.routers.mics import router as mics_router
from cloud.interface.routers.fgis import router as fgis_router
from cloud.interface.routers.workflows import router as workflows_router
from cloud.interface.routers.analytics import router as analytics_router
from cloud.interface.routers.classification import router as classification_router
from cloud.interface.routers.rag import router as rag_router
from cloud.notify.bot_app import start_bot, stop_bot
from cloud.notify.drone_bot_app import start_drone_bot, stop_drone_bot
from cloud.agent.protocol_pdf import generate_protocol
from cloud.agent.rag_agent import query_legal_articles
from cloud.db.incidents import get_incident, update_incident

import httpx
from edge.audio.classifier import AudioResult

logger = logging.getLogger(__name__)

EDGE_CLASSIFY_URL = os.getenv("EDGE_CLASSIFY_URL", "http://edge:8001/api/v1/classify")


def _classify_via_edge(audio_path: str) -> AudioResult:
    """Classify audio by calling edge HTTP API instead of importing TF."""
    import time

    last_err = None
    for attempt in range(2):
        try:
            with open(audio_path, "rb") as f:
                resp = httpx.post(
                    EDGE_CLASSIFY_URL,
                    files={"file": ("audio.wav", f, "audio/wav")},
                    timeout=30.0,
                )
            data = resp.json()
            return AudioResult(
                label=data.get("label", "unknown"),
                confidence=data.get("confidence", 0.0),
                raw_scores=data.get("raw_scores", {}),
            )
        except httpx.ConnectError as e:
            last_err = e
            if attempt == 0:
                logger.warning("Edge not reachable, retrying in 2s...")
                time.sleep(2)
                continue
        except Exception:
            logger.exception("Edge classify API call failed")
            return AudioResult(label="unknown", confidence=0.0, raw_scores={})
    logger.error("Edge classify failed after retry: %s", last_err)
    return AudioResult(label="unknown", confidence=0.0, raw_scores={})


async def _auto_demo():
    """Auto-start a demo scenario after container boot."""
    if os.getenv("DISABLE_AUTO_DEMO"):
        logger.info("Auto-demo disabled by DISABLE_AUTO_DEMO env var")
        return
    delay = random.uniform(45, 60)
    logger.info("Auto-demo scheduled in %.0f seconds (waiting for healthcheck)", delay)
    await asyncio.sleep(delay)

    scenario = random.choice(["chainsaw", "gunshot", "engine"])
    logger.info("Auto-demo: %s", scenario)
    try:
        await _run_demo(scenario)
    except Exception:
        logger.exception("Auto-demo failed")


@asynccontextmanager
async def lifespan(app):
    try:
        await start_bot()
    except Exception:
        logger.exception("Failed to start Ranger bot polling")
    logger.warning("Starting Drone bot...")
    try:
        await start_drone_bot()
    except Exception:
        logger.exception("Failed to start Drone bot polling")
    for attempt in range(3):
        try:
            await asyncio.to_thread(seed_microphones)
            break
        except Exception:
            if attempt < 2:
                logger.warning(
                    "Seed microphones attempt %d failed, retrying in 5s...", attempt + 1
                )
                await asyncio.sleep(5)
            else:
                logger.warning(
                    "Failed to seed microphones after 3 attempts (data may already exist, use POST /api/v1/mics/reseed)"
                )
    asyncio.create_task(_auto_demo())
    yield
    await stop_drone_bot()
    await stop_bot()


app = FastAPI(title="ForestGuard", lifespan=lifespan)
app.include_router(rangers_router)
app.include_router(permits_router)
app.include_router(mics_router)
app.include_router(fgis_router)
app.include_router(workflows_router)
app.include_router(analytics_router)
app.include_router(classification_router)
app.include_router(rag_router)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("unhandled %s on %s", type(exc).__name__, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/incidents/{incident_id}/protocol.pdf")
async def protocol_pdf(incident_id: str):
    incident = get_incident(incident_id)
    if not incident:
        return JSONResponse(status_code=404, content={"error": "incident not found"})

    if incident.protocol_pdf:
        pdf_bytes = incident.protocol_pdf
    else:
        # Try RAG for legal articles, fallback to empty string
        legal_articles = ""
        try:
            legal_articles = await asyncio.wait_for(
                query_legal_articles(incident.audio_class, incident.lat, incident.lon),
                timeout=10,
            )
        except Exception:
            logger.warning(
                "RAG failed for incident %s, generating without legal articles",
                incident_id,
            )

        try:
            pdf_bytes = generate_protocol(incident, legal_articles)
        except Exception:
            logger.exception("PDF generation failed for incident %s", incident_id)
            return JSONResponse(
                status_code=500, content={"error": "PDF generation failed"}
            )
        update_incident(incident_id, protocol_pdf=pdf_bytes)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="protocol_{incident_id}.pdf"'
        },
    )


FRONTEND_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
app.mount("/static/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_clients: list[WebSocket] = []


async def broadcast(event: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _clients.remove(ws)


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text())


@app.get("/analytics")
async def analytics_page():
    """Full-screen DataLens analytics page."""
    return HTMLResponse((FRONTEND_DIR / "analytics.html").read_text())


# ---- REST API v1 — designed for React/Flutter frontends ----


class DemoRequest(BaseModel):
    scenario: str = "chainsaw"
    source_lat: float | None = None
    source_lon: float | None = None


class DemoResponse(BaseModel):
    status: str
    scenario: str


@app.post(
    "/api/v1/demo",
    response_model=DemoResponse,
    dependencies=[Depends(require_api_key)],
)
async def start_demo_v1(req: DemoRequest):
    """Start a demo scenario. Optionally specify source coordinates."""
    asyncio.create_task(
        _run_demo(req.scenario, source_lat=req.source_lat, source_lon=req.source_lon)
    )
    return DemoResponse(status="started", scenario=req.scenario)


# Rangers API — moved to cloud/interface/routers/rangers.py


# Permits API — moved to cloud/interface/routers/permits.py


# RAG, Classification, Analytics, Mics, FGIS-LK, Workflows, Live —
# moved to cloud/interface/routers/


# ---- Gateway event forwarding (uses broadcast directly) ----


class GatewayEvent(BaseModel):
    event: str
    model_config = {"extra": "allow"}


@app.post("/api/v1/gateway-event", dependencies=[Depends(require_api_key)])
async def receive_gateway_event(payload: GatewayEvent):
    """Receive a processed event from the gateway and broadcast to dashboard."""
    await broadcast(payload.model_dump())
    return {"status": "broadcast"}


# Legacy endpoint for backward compatibility
@app.post(
    "/api/v1/incidents/{incident_id}/dispatch-drone",
    dependencies=[Depends(require_api_key)],
)
async def dispatch_drone(incident_id: str):
    """Manually dispatch drone to incident location (VERIFY Telegram button)."""
    incident = get_incident(incident_id)
    if not incident:
        return JSONResponse(status_code=404, content={"error": "incident not found"})

    asyncio.create_task(
        _run_drone_for_incident(
            incident_id, incident.lat, incident.lon, incident.audio_class
        )
    )
    return {"status": "dispatched", "incident_id": incident_id}


async def _run_drone_for_incident(
    incident_id: str, lat: float, lon: float, audio_class: str
) -> None:
    """Run drone pipeline for a manually dispatched VERIFY incident."""
    try:
        from edge.drone.simulated import SimulatedDrone
        from cloud.vision.classifier import classify_photo
        from cloud.agent.decision import compose_alert
        from cloud.notify.telegram import send_confirmed
        from cloud.db.microphones import get_nearest_online

        nearest = get_nearest_online(lat, lon, limit=1)
        home_lat = nearest[0].lat if nearest else lat
        home_lon = nearest[0].lon if nearest else lon

        drone = SimulatedDrone(
            home_lat=home_lat, home_lon=home_lon, scenario=audio_class
        )
        await drone.takeoff()
        async for pos in drone.fly_to(lat, lon):
            await broadcast({"event": "drone_moving", "lat": pos.lat, "lon": pos.lon})
        photo = await drone.capture_photo()
        await broadcast({"event": "drone_photo", "drone_b64": photo.b64})
        await drone.return_home()

        vision_result = await classify_photo(photo.b64)
        alert = await compose_alert(
            audio_class=audio_class,
            visual_description=vision_result.description,
            lat=lat,
            lon=lon,
            confidence=0.0,
            has_human=vision_result.has_human,
            has_fire=vision_result.has_fire,
            has_felling=vision_result.has_felling,
            has_machinery=vision_result.has_machinery,
        )

        incident = get_incident(incident_id)
        if incident:
            import base64 as b64mod

            photo_bytes = b64mod.b64decode(photo.b64) if photo.b64 else None
            await send_confirmed(alert, photo_bytes, incident)

        await broadcast({"event": "pipeline_end", "reason": "manual_drone_complete"})
    except Exception:
        logger.exception("Manual drone dispatch failed for incident %s", incident_id)
        await broadcast({"event": "pipeline_end", "reason": "drone_error"})


@app.post("/demo/start", dependencies=[Depends(require_api_key)])
async def start_demo_legacy(scenario: str = "chainsaw"):
    asyncio.create_task(_run_demo(scenario))
    return {"status": "started", "scenario": scenario}


MIN_DEMO_MEMORY_MB = int(os.getenv("MIN_DEMO_MEMORY_MB", "400"))


def _available_memory_mb() -> float:
    """Available memory in MB. Reads /proc/meminfo (Linux), fallback: inf."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return float("inf")


def _import_demo_deps():
    """Import heavy demo dependencies (TF, simulators). May raise ImportError/MemoryError."""
    from simulator.audio.mic_stream import MicSimulator
    from simulator.drone.drone_stream import DroneSimulator
    from simulator.lora.socket_relay import LoraRelay
    from edge.audio.onset import detect_onset as detect_onset_fn
    from edge.tdoa.triangulate import triangulate, MicPosition
    from edge.decision.decider import decide
    from edge.drone.simulated import SimulatedDrone
    from cloud.vision.classifier import classify_photo
    from cloud.agent.decision import compose_alert
    from cloud.notify.telegram import send_pending, send_confirmed
    from cloud.db.microphones import get_online

    return {
        "MicSimulator": MicSimulator,
        "classify": _classify_via_edge,
        "detect_onset": detect_onset_fn,
        "triangulate": triangulate,
        "MicPosition": MicPosition,
        "decide": decide,
        "SimulatedDrone": SimulatedDrone,
        "classify_photo": classify_photo,
        "compose_alert": compose_alert,
        "send_pending": send_pending,
        "send_confirmed": send_confirmed,
        "get_online": get_online,
    }


async def _run_demo(
    scenario: str,
    source_lat: float | None = None,
    source_lon: float | None = None,
):
    avail = _available_memory_mb()
    if avail < MIN_DEMO_MEMORY_MB:
        logger.warning(
            "Demo skipped: %.0f MB available < %d MB required",
            avail,
            MIN_DEMO_MEMORY_MB,
        )
        await broadcast({"event": "pipeline_end", "reason": "low_memory"})
        return

    try:
        deps = _import_demo_deps()
    except Exception:
        logger.exception("Demo: failed to import dependencies (TF/classifier)")
        await broadcast({"event": "pipeline_end", "reason": "import_error"})
        return

    try:
        MicPosition = deps["MicPosition"]
        import os
        from cloud.db.microphones import random_point_in_boundary, get_nearest_online

        # Generate random source in polygon if not specified
        if source_lat is None or source_lon is None:
            source_lat, source_lon = random_point_in_boundary()

        # Find nearest online mics to the source point
        N_MICS = int(os.getenv("TDOA_N_MICS", "6"))
        online_mics = get_nearest_online(source_lat, source_lon, n=N_MICS)
        if len(online_mics) >= 3:
            mic_positions = [MicPosition(lat=m.lat, lon=m.lon) for m in online_mics]
        else:
            mic_positions = [
                MicPosition(
                    lat=float(os.getenv("MIC_A_LAT", 57.3697)),
                    lon=float(os.getenv("MIC_A_LON", 44.6200)),
                ),
                MicPosition(
                    lat=float(os.getenv("MIC_B_LAT", 57.3752)),
                    lon=float(os.getenv("MIC_B_LON", 44.6345)),
                ),
                MicPosition(
                    lat=float(os.getenv("MIC_C_LAT", 57.3631)),
                    lon=float(os.getenv("MIC_C_LON", 44.6489)),
                ),
            ]

        mic_coords = [(m.lat, m.lon) for m in mic_positions]
        home_lat = mic_positions[0].lat
        home_lon = mic_positions[0].lon

        print(
            f"TDOA: using {len(mic_positions)} mics (nearest to {source_lat:.4f}, {source_lon:.4f})",
            flush=True,
        )
        for i, m in enumerate(mic_positions):
            print(f"  mic[{i}]: {m.lat:.4f}, {m.lon:.4f}", flush=True)

        await broadcast(
            {
                "event": "mic_active",
                "mics": [{"lat": m.lat, "lon": m.lon} for m in mic_positions],
            }
        )
        await broadcast(
            {
                "event": "source_point",
                "lat": source_lat,
                "lon": source_lon,
                "scenario": scenario,
            }
        )
        await asyncio.sleep(0.5)

        mic_sim = deps["MicSimulator"](
            scenario,
            source_lat=source_lat,
            source_lon=source_lon,
            mic_positions=mic_coords,
        )
        signals, audio_paths = await mic_sim.get_signals()
        print(
            f"MicSim: {len(signals)} signals, lengths: {[len(s) for s in signals]}",
            flush=True,
        )

        # Onset detection — pre-filled quiet baseline so gunshot/engine also trigger
        onset = deps["detect_onset"](signals[0])
        await broadcast(
            {
                "event": "onset_check",
                "triggered": onset.triggered,
                "energy_ratio": round(onset.energy_ratio, 2),
            }
        )

        if not onset.triggered:
            await broadcast(
                {
                    "event": "pipeline_end",
                    "reason": f"no_onset (ratio={onset.energy_ratio:.1f})",
                }
            )
            return

        audio_result = deps["classify"](audio_paths[0])

        if audio_result.label in ("background", "unknown") and scenario not in (
            "normal",
            "silence",
        ):
            logger.warning(
                "Classifier returned '%s' for demo scenario '%s' "
                "(confidence=%.2f, scores=%s)",
                audio_result.label,
                scenario,
                audio_result.confidence,
                audio_result.raw_scores,
            )

        await broadcast(
            {
                "event": "audio_classified",
                "class": audio_result.label,
                "confidence": audio_result.confidence,
            }
        )
        await asyncio.sleep(0.3)

        location = deps["triangulate"](signals, mic_positions)
        print(
            f"TDOA result: lat={location.lat:.4f}, lon={location.lon:.4f}, "
            f"error_m={location.error_m:.1f} (source was {source_lat:.4f}, {source_lon:.4f})",
            flush=True,
        )

        decision = deps["decide"](audio_result, location)

        await broadcast(
            {
                "event": "location_found",
                "lat": location.lat,
                "lon": location.lon,
                "error_m": location.error_m,
            }
        )

        await broadcast(
            {
                "event": "agent_decision",
                "send_drone": decision.send_drone,
                "priority": decision.priority,
                "reason": decision.reason,
            }
        )

        if not decision.send_drone:
            if decision.send_lora:
                # VERIFY: record incident + notify rangers (no drone photo)
                await deps["send_pending"](
                    location.lat,
                    location.lon,
                    audio_result.label,
                    decision.reason,
                    confidence=audio_result.confidence,
                    is_demo=True,
                    broadcast=True,
                )
                await broadcast({"event": "pipeline_end", "reason": "verify_no_drone"})
            else:
                await broadcast({"event": "pipeline_end", "reason": "no_anomaly"})
            return

        drone = deps["SimulatedDrone"](
            home_lat=home_lat, home_lon=home_lon, scenario=scenario
        )
        await drone.takeoff()

        async def drone_task():
            async for pos in drone.fly_to(location.lat, location.lon):
                await broadcast(
                    {"event": "drone_moving", "lat": pos.lat, "lon": pos.lon}
                )
            photo = await drone.capture_photo()
            await broadcast({"event": "drone_photo", "drone_b64": photo.b64})
            return photo

        # send_pending creates an Incident and returns it
        photo, incident = await asyncio.gather(
            drone_task(),
            deps["send_pending"](
                location.lat,
                location.lon,
                audio_result.label,
                decision.reason,
                confidence=audio_result.confidence,
                is_demo=True,
                broadcast=True,
            ),
        )

        vision_result = await deps["classify_photo"](photo.b64)
        await broadcast(
            {
                "event": "vision_classified",
                "description": vision_result.description,
                "has_human": vision_result.has_human,
                "has_fire": vision_result.has_fire,
                "has_felling": vision_result.has_felling,
                "has_machinery": vision_result.has_machinery,
                "is_threat": vision_result.is_threat,
                "time_of_day": vision_result.time_of_day,
                "people_count": vision_result.people_count,
                "equipment_types": vision_result.equipment_types,
                "vegetation_damage": vision_result.vegetation_damage,
                "damage_area_estimate": vision_result.damage_area_estimate,
            }
        )

        alert = await deps["compose_alert"](
            audio_class=audio_result.label,
            visual_description=vision_result.description,
            lat=location.lat,
            lon=location.lon,
            confidence=audio_result.confidence,
            has_human=vision_result.has_human,
            has_fire=vision_result.has_fire,
            has_felling=vision_result.has_felling,
            has_machinery=vision_result.has_machinery,
        )
        # Store drone photo in incident (sent to ranger after accept)
        await deps["send_confirmed"](alert, photo.data, incident=incident)
        await broadcast(
            {
                "event": "alert_sent",
                "text": alert.text,
                "priority": alert.priority,
                "incident_id": incident.id,
            }
        )
        await drone.return_home()
        await broadcast({"event": "pipeline_end", "reason": "complete"})
    except Exception:
        logger.exception("Demo pipeline error")
        await broadcast({"event": "pipeline_end", "reason": "error"})
