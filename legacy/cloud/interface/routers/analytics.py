"""DataLens analytics and export API router."""

import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from cloud.interface.security import require_api_key

router = APIRouter()


@router.get(
    "/api/v1/incidents/export",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
)
async def export_incidents_csv():
    """Export incidents as CSV for DataLens integration."""
    from cloud.analytics.datalens import get_datalens_incidents

    rows = get_datalens_incidents()
    if not rows:
        return PlainTextResponse(content="", media_type="text/csv")

    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=incidents.csv"},
    )


@router.get("/api/v1/ai-studio-stack")
async def ai_studio_stack():
    """Show all Yandex Cloud AI Studio integrations used."""
    return {
        "integrations": [
            {
                "service": "YandexGPT",
                "usage": "Alert composition, legal text generation",
            },
            {
                "service": "AI Studio Assistants API",
                "usage": "RAG agent with File Search",
            },
            {
                "service": "File Search (RAG)",
                "usage": "Legal knowledge base (9 normative docs)",
            },
            {
                "service": "Web Search",
                "usage": "Real-time legal updates from consultant.ru/garant.ru",
            },
            {"service": "SpeechKit STT", "usage": "Voice message transcription"},
            {"service": "YandexGPT Vision", "usage": "Drone photo analysis"},
            {
                "service": "DataSphere",
                "usage": "ML model training & deployment (YAMNet)",
            },
            {"service": "DataLens", "usage": "Analytics dashboard for management"},
            {
                "service": "Yandex Workflows",
                "usage": "Incident processing pipeline orchestration",
            },
        ]
    }


@router.get("/api/v1/datalens/incidents", dependencies=[Depends(require_api_key)])
async def datalens_incidents():
    """JSON incidents data for DataLens API connector."""
    from cloud.analytics.datalens import get_datalens_incidents

    return get_datalens_incidents()
