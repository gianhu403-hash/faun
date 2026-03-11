import httpx
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
# TODO: Yandex Foundation Models completions API
# Docs: https://yandex.cloud/docs/foundation-models/
API_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

SYSTEM_PROMPT = """
Ты — AI-система акустического мониторинга леса Faun.
Получаешь данные от акустических датчиков и камеры дрона.
Твоя задача: написать чёткий, КОНКРЕТНЫЙ алерт егерю.

Правила:
- Пиши по-русски, кратко, по существу
- Укажи ТИП УГРОЗЫ и конкретные визуальные признаки
- Если обнаружены люди — укажи и рекомендуй осторожность
- Если обнаружена техника — укажи тип и масштаб
- Если обнаружен огонь — приоритет: безопасность инспектора
- Дай 1-2 КОНКРЕТНЫХ действия (не «проверить место»)
- Координаты уже указаны в карточке — НЕ дублируй
- 3-4 предложения максимум
"""


@dataclass
class Alert:
    text: str
    priority: str
    lat: float
    lon: float


async def compose_alert(
    audio_class: str,
    visual_description: str,
    lat: float,
    lon: float,
    confidence: float,
    has_human: bool = False,
    has_fire: bool = False,
    has_felling: bool = False,
    has_machinery: bool = False,
) -> Alert:

    priority_map = {
        "chainsaw": "ВЫСОКИЙ",
        "gunshot": "ВЫСОКИЙ",
        "fire": "ВЫСОКИЙ",
        "unknown": "СРЕДНИЙ",
    }
    priority = priority_map.get(audio_class, "СРЕДНИЙ")

    visual_details = []
    if has_human:
        visual_details.append("обнаружены люди")
    if has_fire:
        visual_details.append("открытый огонь")
    if has_felling:
        visual_details.append("следы/процесс рубки")
    if has_machinery:
        visual_details.append("тяжёлая техника")

    visual_section = visual_description
    if visual_details:
        visual_section += "\nВизуальные признаки: " + "; ".join(visual_details)

    prompt = f"""
Данные с датчиков:
- Звук: {audio_class} (уверенность {confidence:.0%})
- Визуальный анализ дрона: {visual_section}
- Координаты: {lat:.4f}°N, {lon:.4f}°E
- Приоритет: {priority}

Напиши алерт егерю.
"""
    text = await _call_yandex(prompt)
    return Alert(text=text, priority=priority, lat=lat, lon=lon)


async def _call_yandex(user_prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                API_URL,
                headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
                json={
                    "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
                    "completionOptions": {
                        "stream": False,
                        "temperature": 0.2,
                        "maxTokens": 350,
                    },
                    "messages": [
                        {"role": "system", "text": SYSTEM_PROMPT},
                        {"role": "user", "text": user_prompt},
                    ],
                },
            )
        resp.raise_for_status()
        return resp.json()["result"]["alternatives"][0]["message"]["text"]
    except Exception:
        logger.exception("YandexGPT call failed")
        return "Обнаружено нарушение. Требуется проверка инспектором на месте."
