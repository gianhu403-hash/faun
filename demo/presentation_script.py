"""Presentation demo script — 5-step demo flow for jury.

Usage (on VPS):
    python -m demo.presentation_script

Or step by step:
    python -m demo.presentation_script --step 1
    python -m demo.presentation_script --step 2
    ...

Steps:
  1. System health check (docker, endpoints, Telegram bot)
  2. Dashboard — open Leaflet map with mic grid
  3. Live pipeline — chainsaw → TDOA → alert → Telegram
  4. RAG agent — legal question via Telegram
  5. AI Studio introspection — show all 12 integrations
"""

import argparse
import asyncio
import json
import sys

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 15


def _header(step: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  ШАГ {step}: {title}")
    print(f"{'=' * 60}\n")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [!!] {msg}")


def _info(msg: str) -> None:
    print(f"  [..] {msg}")


async def step_1_health():
    """Check system health."""
    _header(1, "ПРОВЕРКА СИСТЕМЫ")

    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        # Health endpoint
        try:
            r = await c.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                _ok("Cloud service: healthy")
            else:
                _fail(f"Cloud service: {r.status_code}")
        except Exception as e:
            _fail(f"Cloud service: {e}")

        # AI Studio introspection
        try:
            r = await c.get(f"{BASE_URL}/api/v1/ai-studio")
            data = r.json()
            total = data.get("total_integrations", "?")
            active = data.get("active_integrations", "?")
            _ok(f"AI Studio: {active}/{total} интеграций активны")
        except Exception as e:
            _fail(f"AI Studio endpoint: {e}")

        # Rangers
        try:
            r = await c.get(f"{BASE_URL}/api/v1/rangers")
            rangers = r.json()
            _ok(f"Егерей зарегистрировано: {len(rangers)}")
        except Exception as e:
            _fail(f"Rangers endpoint: {e}")


async def step_2_dashboard():
    """Show dashboard info."""
    _header(2, "ДАШБОРД (LEAFLET)")
    _info(f"Откройте в браузере: {BASE_URL}")
    _info("Покажите жюри:")
    _info("  - Карта с микрофонами (зелёные маркеры)")
    _info("  - Зоны патрулирования (полигоны)")
    _info("  - Панель статистики справа")
    _info("  - WebSocket подключение (алерты в реальном времени)")


async def step_3_live_pipeline():
    """Run chainsaw demo scenario."""
    _header(3, "LIVE PIPELINE: БЕНЗОПИЛА")

    async with httpx.AsyncClient(timeout=30) as c:
        _info("Запуск сценария: chainsaw...")
        try:
            r = await c.post(
                f"{BASE_URL}/api/v1/demo",
                json={"scenario": "chainsaw"},
            )
            data = r.json()
            _ok(f"Сценарий запущен: {data}")
            _info("Смотрите:")
            _info("  1. Карта — новый алерт (красный маркер)")
            _info("  2. Telegram @ya_faun_bot — алерт с кнопкой 'Принять вызов'")
            _info("  3. WebSocket — событие в панели логов")
            _info("")
            _info("Подождите ~5 секунд для полного pipeline...")
        except Exception as e:
            _fail(f"Demo start failed: {e}")


async def step_4_rag_agent():
    """Test RAG agent."""
    _header(4, "RAG-АГЕНТ")
    _info("В Telegram @ya_faun_bot отправьте:")
    _info('  "Какие штрафы за незаконную рубку леса?"')
    _info("")
    _info("Бот ответит на основе 9 нормативных документов:")
    _info("  - УК РФ, КоАП РФ, Лесной кодекс")
    _info("  - Приказы Минприроды, Рослесхоза")
    _info("")
    _info("Также можно отправить голосовое сообщение (SpeechKit STT)")


async def step_5_ai_studio():
    """Show AI Studio integrations."""
    _header(5, "YANDEX CLOUD AI STUDIO")

    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            r = await c.get(f"{BASE_URL}/api/v1/ai-studio")
            data = r.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            _fail(f"AI Studio: {e}")

    _info("")
    _info("Ключевые интеграции для жюри:")
    _info("  1. YandexGPT — генерация алертов, юридизация")
    _info("  2. Assistants API + File Search — RAG по 9 документам")
    _info("  3. Web Search — актуальные правовые нормы")
    _info("  4. SpeechKit STT — голосовые сообщения")
    _info("  5. Gemma 3 27B — анализ фото с дрона")
    _info("  6. DataSphere — обучение YAMNet v7")
    _info("  7. DataLens — аналитический дашборд")
    _info("  8. Yandex Workflows — 12-шаговый pipeline")


STEPS = {
    1: step_1_health,
    2: step_2_dashboard,
    3: step_3_live_pipeline,
    4: step_4_rag_agent,
    5: step_5_ai_studio,
}


async def run_all():
    for step_fn in STEPS.values():
        await step_fn()
        print()

    print("\n" + "=" * 60)
    print("  ДЕМО ЗАВЕРШЕНО")
    print("=" * 60)
    print("\nДополнительно можно показать:")
    print("  - Jupyter ноутбуки: docs/notebooks/")
    print("  - ML метрики: docs/results/")
    print("  - PDF протокол: через Telegram бота")


def main():
    parser = argparse.ArgumentParser(description="Presentation demo script")
    parser.add_argument(
        "--step", type=int, choices=range(1, 6), help="Run specific step"
    )
    args = parser.parse_args()

    if args.step:
        asyncio.run(STEPS[args.step]())
    else:
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
