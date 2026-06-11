"""RAG legal query API router."""

import asyncio
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cloud.agent.rag_agent import query_rag, query_rag_enriched, IncidentContext
from cloud.interface.security import require_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

CLASS_FALLBACK_ARTICLES = {
    "chainsaw": "Ст. 260 УК РФ (ч. 1-3), ст. 8.28 КоАП РФ, ст. 96 ЛК РФ",
    "axe": "Ст. 260 УК РФ (ч. 1-3), ст. 8.28 КоАП РФ, ст. 96 ЛК РФ",
    "gunshot": "Ст. 258 УК РФ, ст. 8.35 КоАП РФ",
    "fire": "Ст. 261 УК РФ, ст. 8.32 КоАП РФ",
    "engine": "Ст. 8.25 КоАП РФ, ст. 96 ЛК РФ",
}
DEFAULT_FALLBACK_ARTICLES = "Ст. 260 УК РФ, ст. 8.28 КоАП РФ, ст. 96 ЛК РФ"


class RagQueryRequest(BaseModel):
    question: str
    context: str = ""
    # Structured incident fields (optional, for enriched RAG)
    audio_class: str | None = None
    confidence: float | None = None
    lat: float | None = None
    lon: float | None = None
    vision_description: str | None = None
    has_felling: bool | None = None
    has_human: bool | None = None
    has_fire: bool | None = None
    has_machinery: bool | None = None
    people_count: int | None = None
    equipment_types: list[str] | None = None
    vegetation_damage: str | None = None
    damage_area_estimate: str | None = None


class RagQueryResponse(BaseModel):
    answer: str


@router.post(
    "/api/v1/rag-query",
    response_model=RagQueryResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_query_endpoint(req: RagQueryRequest):
    """Query RAG agent with File Search + Web Search (Yandex AI Studio)."""
    try:
        if req.audio_class:
            ctx = IncidentContext(
                audio_class=req.audio_class,
                confidence=req.confidence or 0.0,
                lat=req.lat or 0.0,
                lon=req.lon or 0.0,
                vision_description=req.vision_description or "",
                has_felling=req.has_felling or False,
                has_human=req.has_human or False,
                has_fire=req.has_fire or False,
                has_machinery=req.has_machinery or False,
                people_count=req.people_count or 0,
                equipment_types=req.equipment_types or [],
                vegetation_damage=req.vegetation_damage or "",
                damage_area_estimate=req.damage_area_estimate or "",
            )
            answer = await asyncio.wait_for(query_rag_enriched(ctx), timeout=25)
        else:
            answer = await asyncio.wait_for(
                query_rag(req.question, req.context), timeout=25
            )
    except asyncio.TimeoutError:
        logger.warning("RAG query timed out after 25s")
        articles = CLASS_FALLBACK_ARTICLES.get(
            req.audio_class, DEFAULT_FALLBACK_ARTICLES
        )
        answer = (
            "Превышено время ожидания ответа от YandexGPT.\n\n"
            f"## ПРАВОВАЯ БАЗА\n{articles}\n\n"
            "## КВАЛИФИКАЦИЯ\nТребуется детальный анализ после восстановления связи с YandexGPT.\n\n"
            "## ДЕЙСТВИЯ ИНСПЕКТОРА\n"
            "1. Оцените обстановку, не приближайтесь в одиночку\n"
            "2. Зафиксируйте GPS-координаты\n"
            "3. Сделайте фото/видео нарушения\n"
            "4. Не вступайте в конфликт с нарушителями\n"
            "5. Вызовите патрульную группу\n"
            "6. Составьте акт по форме (ст. 96 ЛК РФ)"
        )
    except Exception as e:
        logger.error("RAG query error: %s", e)
        answer = (
            "Ошибка при запросе к YandexGPT.\n\n"
            "## ДЕЙСТВИЯ ИНСПЕКТОРА\n"
            "1. Зафиксируйте GPS-координаты\n"
            "2. Сделайте фото/видео нарушения\n"
            "3. Вызовите патрульную группу\n"
            "4. Составьте акт по форме (ст. 96 ЛК РФ)"
        )
    return RagQueryResponse(answer=answer)
