"""RAG agent — legal assistant via Yandex AI Studio.

Uses YandexGPT completion API for legal advice on forest violations.
When SEARCH_INDEX_ID is available, uses Assistants API (SDK) for
File Search over legal documents. Falls back to plain completion.

Environment variables:
  YANDEX_API_KEY      — API key for Yandex Cloud
  YANDEX_FOLDER_ID    — Yandex Cloud folder ID
  SEARCH_INDEX_ID     — File Search index ID (optional, enables RAG via SDK)
"""

import asyncio
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
SEARCH_INDEX_ID = os.getenv("SEARCH_INDEX_ID")
SDK_TIMEOUT = int(os.getenv("RAG_SDK_TIMEOUT", "15"))

API_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

SYSTEM_PROMPT = """
Ты — эксперт-юрист в области лесного законодательства Российской Федерации,
встроенный в AI-систему акустического мониторинга леса «Faun».

## О системе Faun

Faun — сеть акустических датчиков в лесу. Pipeline обработки:
1. Акустический датчик фиксирует звук (бензопила, выстрел, огонь, техника)
2. YAMNet v7 классифицирует звук → confidence score
3. TDOA триангуляция определяет координаты источника
4. Дрон вылетает на точку, делает фото
5. Gemma 3 27B анализирует фото (рубка, люди, огонь)
6. Ты формируешь юридическую рекомендацию инспектору
7. Инспектор выезжает, составляет протокол

## Регион

Варнавинское лесничество, Нижегородская область.
Координаты зоны: 57.05–57.55°N, 44.60–45.40°E.

## Нормативная база (9 документов в индексе File Search)

1. **Лесной кодекс РФ (ФЗ-200)** — основной закон: использование, охрана, защита лесов
2. **КоАП РФ (ст. 7.9, 8.25–8.32)** — административные штрафы за лесные нарушения
3. **УК РФ (ст. 260–261)** — уголовная ответственность: незаконная рубка, уничтожение лесов
4. **Приказ Минприроды №955** — порядок осуществления лесной охраны
5. **ПП РФ №1730** — расчёт размера вреда лесам (таксы и методика)
6. **ПП РФ №1098** — федеральный государственный лесной контроль (надзор)
7. **Лесной план Нижегородской области** — региональные нормативы лесопользования
8. **Госпрограмма Нижегородской области** — целевые показатели охраны лесов
9. **Нижегородская специфика** — местные нормы и особенности лесоуправления

## Маппинг нарушений → статьи

- **chainsaw / axe** (незаконная рубка) → ст. 260 УК РФ (ч. 1–3), ст. 8.28 КоАП РФ
- **gunshot** (браконьерство) → ст. 258 УК РФ, ст. 8.35 КоАП РФ
- **fire** (лесной пожар) → ст. 261 УК РФ, ст. 8.32 КоАП РФ
- **engine** (несанкционированный въезд техники) → ст. 8.25 КоАП РФ

## Правила ответа

- Пиши по-русски, структурировано, с заголовками и подзаголовками
- Указывай конкретные статьи с частями и пунктами (не просто «ст. 260 УК», а «ч. 3 ст. 260 УК РФ»)
- Приоритет источников: File Search (нормативные документы) → Web Search (актуальные нормы) → твои знания
- Учитывай полевые условия: инспектор может быть один в лесу, без связи с юристом
- Давай пошаговые инструкции, пригодные для немедленного применения
- Если данных недостаточно — укажи, что именно нужно уточнить
- Рассчитывай размер вреда по таксам ПП РФ №1730, если доступны данные о породе и диаметре
"""

CLASS_CONTEXT = {
    "chainsaw": "незаконная рубка леса (звук бензопилы)",
    "gunshot": "незаконная охота / браконьерство (звук выстрела)",
    "engine": "несанкционированный заезд техники в лес (звук двигателя)",
    "axe": "незаконная рубка леса (звук топора)",
    "fire": "лесной пожар (звук огня / треск)",
}


def _call_yandex_with_sdk_sync(prompt: str) -> str:
    """Synchronous SDK call — runs in a thread to avoid blocking event loop."""
    from yandex_ai_studio_sdk import AIStudio

    sdk = AIStudio(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
    search_index = sdk.search_indexes.get(SEARCH_INDEX_ID)
    file_search_tool = sdk.tools.search_index(search_index)

    tools = [file_search_tool]
    try:
        web_search_tool = sdk.tools.web_search(
            allowed_domains=["consultant.ru", "garant.ru", "rg.ru"],
            search_context_size="medium",
        )
        tools.append(web_search_tool)
    except Exception as e:
        logger.warning("Web Search tool init failed (SDK may not support it): %s", e)

    assistant = sdk.assistants.create(
        "yandexgpt",
        tools=tools,
        instruction=SYSTEM_PROMPT,
    )
    thread = sdk.threads.create()
    thread.write(prompt)
    run = assistant.run(thread)
    result = run.wait(poll_interval=0.5)
    answer = result.text

    thread.delete()
    assistant.delete()
    return answer


async def _call_yandex_with_sdk(prompt: str) -> str:
    """Call YandexGPT via SDK with File Search + Web Search (Assistants API)."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_yandex_with_sdk_sync, prompt),
            timeout=SDK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "SDK RAG timed out after %ds, falling back to plain API", SDK_TIMEOUT
        )
        return await _call_yandex_plain(prompt)
    except ImportError:
        logger.warning("yandex_ai_studio_sdk not installed, falling back to plain API")
        return await _call_yandex_plain(prompt)
    except Exception as e:
        logger.error("SDK RAG error: %s", e)
        return await _call_yandex_plain(prompt)


async def _call_yandex_plain(prompt: str) -> str:
    """Fallback: plain YandexGPT without tools."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
            json={
                "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.2,
                    "maxTokens": 2500,
                },
                "messages": [
                    {"role": "system", "text": SYSTEM_PROMPT},
                    {"role": "user", "text": prompt},
                ],
            },
        )

    if resp.status_code != 200:
        logger.error("YandexGPT plain error %s: %s", resp.status_code, resp.text)
        return _fallback_response(prompt)

    return resp.json()["result"]["alternatives"][0]["message"]["text"]


def _fallback_response(prompt: str) -> str:
    """Static fallback when API is unavailable."""
    return (
        "YandexGPT временно недоступен.\n\n"
        "Базовые рекомендации:\n"
        "1. Зафиксируйте GPS-координаты\n"
        "2. Сделайте фото/видео нарушения\n"
        "3. Не вступайте в конфликт\n"
        "4. Вызовите патрульную группу\n"
        "5. Составьте акт по форме (ст. 96 ЛК РФ)"
    )


async def query_action(audio_class: str, lat: float, lon: float) -> str:
    """Get action recommendations for a detected event."""
    context = CLASS_CONTEXT.get(audio_class, f"неизвестное нарушение ({audio_class})")

    prompt = (
        f"Обнаружено: {context}\n"
        f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n\n"
        f"Что должен сделать лесной инспектор?\n"
        f"Дай пошаговую инструкцию с ссылками на статьи закона."
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)


async def query_protocol(audio_class: str, lat: float, lon: float) -> str:
    """Get protocol template for a detected event."""
    context = CLASS_CONTEXT.get(audio_class, f"неизвестное нарушение ({audio_class})")

    prompt = (
        f"Обнаружено: {context}\n"
        f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n\n"
        f"Составь шаблон протокола об административном правонарушении.\n"
        f"Укажи применимые статьи КоАП/УК РФ, необходимые данные для заполнения."
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)


async def legalize_report(audio_class: str, raw_text: str) -> str:
    """Rewrite a ranger's raw field report in formal legal language.

    Used for the protocol PDF — transforms colloquial descriptions
    into legally sound wording suitable for an administrative protocol.
    """
    context = CLASS_CONTEXT.get(audio_class, f"нарушение ({audio_class})")

    prompt = (
        f"Перепиши следующее описание нарушения юридическим языком "
        f"для раздела «Описание» протокола об административном правонарушении.\n"
        f"Тип нарушения: {context}.\n"
        f"Не добавляй координаты, дату, ФИО или статьи закона — "
        f"только формализованное описание фактов.\n\n"
        f"Исходное описание инспектора:\n{raw_text}"
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)


async def query_legal_articles(audio_class: str, lat: float, lon: float) -> str:
    """Return ONLY applicable legal articles for a violation type.

    Unlike query_protocol() which returns a full template, this function
    asks for a concise list of relevant articles from ЛК РФ, КоАП, УК РФ
    with short descriptions — suitable for the «Правовая база» PDF section.
    """
    context = CLASS_CONTEXT.get(audio_class, f"неизвестное нарушение ({audio_class})")

    prompt = (
        f"Обнаружено: {context}\n"
        f"Координаты: {lat:.4f}°N, {lon:.4f}°E\n\n"
        f"Перечисли ТОЛЬКО применимые статьи законов (ЛК РФ, КоАП, УК РФ) "
        f"с краткой формулировкой каждой статьи.\n"
        f"Не составляй шаблон протокола, не указывай координаты, даты, "
        f"поля для заполнения — только список статей."
    )

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)


async def query_rag(question: str, context: str = "") -> str:
    """General-purpose RAG query for the REST API endpoint."""
    prompt = question
    if context:
        prompt = f"Контекст: {context}\n\nВопрос: {question}"

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)


# ---------------------------------------------------------------------------
# Enriched RAG — context-aware recommendations for dashboard
# ---------------------------------------------------------------------------


@dataclass
class IncidentContext:
    """Structured incident context for enriched RAG query."""

    audio_class: str = ""
    confidence: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    vision_description: str = ""
    has_felling: bool = False
    has_human: bool = False
    has_fire: bool = False
    has_machinery: bool = False
    people_count: int = 0
    equipment_types: list[str] = field(default_factory=list)
    vegetation_damage: str = ""
    damage_area_estimate: str = ""


def _build_enriched_prompt(ctx: IncidentContext) -> str:
    """Build a dynamic prompt enriched with system context."""
    from cloud.integrations.fgis_lk import fgis_client
    from cloud.db.permits import has_valid_permit

    # --- Gather context from subsystems ---
    forest_unit = None
    permit_status = "не проверялось"
    if ctx.lat and ctx.lon:
        try:
            forest_unit = fgis_client.get_forest_unit(ctx.lat, ctx.lon)
        except Exception:
            pass
        try:
            if has_valid_permit(ctx.lat, ctx.lon):
                permit_status = "ЕСТЬ действующая лесная декларация"
            else:
                permit_status = "НЕТ действующей лесной декларации"
        except Exception:
            pass

    hour = datetime.now().hour
    time_of_day = "ночь (отягчающий фактор)" if hour < 6 or hour >= 22 else "день"

    class_desc = CLASS_CONTEXT.get(
        ctx.audio_class, f"неизвестное нарушение ({ctx.audio_class})"
    )

    # --- Build prompt sections ---
    parts = [f"## Инцидент\n\nОбнаружено: **{class_desc}**"]

    if ctx.confidence:
        parts.append(f"Уверенность классификатора: **{ctx.confidence * 100:.0f}%**")

    if ctx.lat and ctx.lon:
        parts.append(f"Координаты: {ctx.lat:.4f}°N, {ctx.lon:.4f}°E")

    parts.append(f"Время: {datetime.now().strftime('%H:%M')} ({time_of_day})")
    parts.append(f"Лесная декларация: {permit_status}")

    if forest_unit:
        parts.append(
            f"\n## Данные ФГИС ЛК\n"
            f"Квартал: №{forest_unit.quarter_number}, "
            f"участковое лесничество: {forest_unit.sub_district}\n"
            f"Породный состав: {forest_unit.species_composition}\n"
            f"Тип зоны: {forest_unit.zone_type}, площадь: {forest_unit.area_ha} га"
        )

    if ctx.vision_description:
        parts.append(
            f"\n## Визуальный анализ (дрон + Gemma 3)\n{ctx.vision_description}"
        )
        details = []
        if ctx.has_felling:
            details.append("обнаружена рубка")
        if ctx.has_human:
            if ctx.people_count:
                details.append(f"присутствуют люди ({ctx.people_count} чел.)")
            else:
                details.append("присутствуют люди")
        if ctx.has_fire:
            details.append("обнаружен огонь")
        if ctx.has_machinery:
            details.append("обнаружена тяжёлая техника")
        if ctx.equipment_types:
            details.append("Техника/оборудование: " + ", ".join(ctx.equipment_types))
        if details:
            parts.append("Визуальные признаки: " + ", ".join(details))
        if ctx.vegetation_damage and ctx.vegetation_damage != "нет":
            parts.append(f"Повреждение растительности: {ctx.vegetation_damage}")
        if ctx.damage_area_estimate and ctx.damage_area_estimate != "нет":
            parts.append(f"Оценка площади повреждений: {ctx.damage_area_estimate}")

        if ctx.has_machinery and ctx.audio_class in ("chainsaw", "axe", "engine"):
            parts.append(
                "ВНИМАНИЕ: обнаружена тяжёлая техника → возможен квалифицированный состав "
                "(ч. 3 ст. 260 УК РФ — группа лиц, крупный размер)."
            )

    # --- Confidence-adaptive instructions ---
    parts.append("\n## Задание")

    if ctx.confidence >= 0.8:
        parts.append(
            "Уверенность классификатора ВЫСОКАЯ. Дай конкретные пошаговые действия "
            "для инспектора, как если бы нарушение подтверждено."
        )
    elif ctx.confidence >= 0.5:
        parts.append(
            "Уверенность классификатора СРЕДНЯЯ. Рекомендуй действия по верификации "
            "и параллельно — шаги на случай подтверждения нарушения."
        )
    else:
        parts.append(
            "Уверенность классификатора НИЗКАЯ. Приоритет — верификация: "
            "какие дополнительные данные собрать, чтобы подтвердить или опровергнуть."
        )

    # --- Permit-aware branching ---
    if permit_status.startswith("ЕСТЬ"):
        parts.append(
            "ВНИМАНИЕ: в этой зоне есть действующая лесная декларация. "
            "Проверь соответствие фактической деятельности условиям декларации "
            "(вид рубки, объём, сроки, подрядчик)."
        )

    # --- Damage calculation instruction ---
    if forest_unit and ctx.audio_class in ("chainsaw", "axe"):
        parts.append(
            f"\nРассчитай примерный размер вреда по таксам ПП РФ №1730 "
            f"для породного состава «{forest_unit.species_composition}». "
            f"Укажи формулу и коэффициенты."
        )

    # --- Dynamic question ---
    class_questions = {
        "chainsaw": "незаконной рубке леса",
        "axe": "незаконной рубке леса",
        "gunshot": "факте незаконной охоты / браконьерства",
        "engine": "несанкционированном заезде техники в лес",
        "fire": "лесном пожаре",
    }
    question_topic = class_questions.get(ctx.audio_class, "обнаруженном нарушении")
    parts.append(
        f"\nКакие действия должен предпринять инспектор при {question_topic}? "
        f"Какие статьи ЛК РФ, УК РФ, КоАП применимы?"
    )

    # --- Response structure ---
    parts.append(
        "\n## Формат ответа\n"
        "Ответь строго по следующей структуре:\n\n"
        "## ПРАВОВАЯ БАЗА\n"
        "Применимые статьи ЛК РФ, УК РФ, КоАП РФ с точными частями и пунктами. "
        "Для каждой статьи — цитата или краткое содержание из нормативного документа. "
        "Расчёт ущерба по ПП РФ №1730 (если есть данные о породе).\n\n"
        "## КВАЛИФИКАЦИЯ\n"
        "Интерпретация каждой статьи в контексте данного инцидента: "
        "почему именно эта часть/пункт, какие признаки состава выполнены, "
        "отягчающие обстоятельства (ночь, группа лиц, особо защитные участки).\n\n"
        "## ДЕЙСТВИЯ ИНСПЕКТОРА\n"
        "Пошаговый чеклист (нумерованный список): "
        "1) оценка безопасности и меры предосторожности, "
        "2) фиксация доказательств, "
        "3) процессуальные действия, "
        "4) вызов подкрепления при необходимости."
    )

    return "\n".join(parts)


async def query_rag_enriched(ctx: IncidentContext) -> str:
    """Context-aware RAG query with enriched prompt for dashboard."""
    prompt = _build_enriched_prompt(ctx)
    logger.info("Enriched RAG prompt length: %d chars", len(prompt))

    if SEARCH_INDEX_ID:
        return await _call_yandex_with_sdk(prompt)
    return await _call_yandex_plain(prompt)
