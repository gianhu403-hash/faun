# Faun — Code Audit Roadmap

Зафиксировано 2026-04-07 в ходе подготовки к деплою на shared VPS `delphi-press`. Обновлено 2026-04-07 после split main.py. Цель документа — собрать находки аудита в одном месте, чтобы в будущих сессиях по ним можно было идти точечно.

## Базовые метрики проекта (обновлено после split)

- **Python код**: 118 файлов, ~22 300 строк (8 новых роутеров + тест)
- **cloud/**: `interface/main.py` 639 строк (было 1 307), `interface/routers/` 735 строк (8 файлов)
- **edge/**: 1 748 строк + ML модель `yamnet_forest_classifier_v7.keras` (6.4 MB)
- **gateway/**: 348 строк
- **tests/**: ~8 700 строк, 33 файла, 494 теста (было 453)
- **Финальный docker-образ**: ~1.8–2.0 GB (`python:3.11-slim` + `texlive-*` + `requirements.txt` ≈ 793 MB установленных пакетов)
- **RAM при старте**: cloud ≈ 440 MB, edge ≈ 955 MB (TF + YAMNet), lora_gateway ≈ 50 MB → суммарно ≈ 1.4 GB

## Quick wins (низкая стоимость, заметный эффект)

| # | Изменение | Эффект | Статус |
|---|---|---|---|
| 1 | ~~Убрать `texlive-*` из `Dockerfile`~~ | — | **Оставляем** — texlive нужен для PDF-протоколов |
| 2 | Multi-stage build (builder → runtime) | −500 MB финального образа | **TODO** |
| 3 | Pin upper bounds зависимостей: `tensorflow<2.16`, `numpy<2.0`, `scipy<1.13` | Стабильность production | **TODO** |
| 4 | Отдельные `Dockerfile` для cloud / edge / gateway | Gateway образ −500 MB | **TODO** |
| 5 | ~~Удалить `cloud/integrations/fgis_lk.py`~~ | — | **Отменено** — используется: 3 REST эндпоинта + RAG agent enrichment + тесты |
| 6 | ~~Удалить `cloud/workflows/yandex_workflows.py`~~ | — | **Отменено** — в критическом пути `POST /api/v1/workflow/run` |
| 7 | Закрыть TODO в `cloud/agent/decision.py:11` и `cloud/interface/main.py:214` | Чистота кода | **TODO** |

## Средние улучшения (требуют рефакторинга, но без архитектурных рисков)

| # | Изменение | Что даёт | Статус |
|---|---|---|---|
| 1 | Split `cloud/interface/main.py` на FastAPI routers | Проще навигация и тесты | **Сделано** — 1 307 → 639 LOC, 8 роутеров в `cloud/interface/routers/` |
| 2 | Split `cloud/notify/bot_handlers.py` (1 038 строк) на handlers по доменам | То же | **TODO** — план: bot_core, ranger_registration, ranger_commands, incident_workflow, evidence_collection, protocol_generation, alert_management |
| 3 | Унифицировать `ranger_bot` и `drone_bot` | Меньше дублирования в `cloud/notify/` | **TODO** |
| 4 | Покрыть тестами split-части | Не уронить функциональность при split | **Частично** — 41 route registry тест для main.py split |

## Серьёзные изменения (под отдельный пилот, нужно дополнительное решение)

| # | Изменение | Эффект | Статус |
|---|---|---|---|
| 1 | ~~TensorFlow → TFLite в edge~~ | — | **Отменено** — при 78.6% accuracy потеря 5–15% от TF Hub ops неприемлема. Hybrid (TFLite только head) экономит лишь ~200 MB — не стоит риска |
| 2 | Перевести cloud на YDB-only, удалить SQLite-impl (~1 300 строк) | −1 300 строк, единый DB-бэкенд | **TODO** — нужен local-YDB или эмулятор |
| 3 | Заменить `python-telegram-bot[job-queue]` на aiogram | Меньше зависимостей | **TODO** — полный rewrite handlers |
| 4 | Вынести модель YAMNet в S3 / HF и качать на старте | Не таскать `*.keras` через scp | **TODO** |

## Архитектурные наблюдения (для контекста)

- **Дублирование SQLite/YDB в `cloud/db/`** оправдано factory pattern (`db/factory.py`): локально SQLite, в production YDB. Удалять только вместе с переходом на YDB-only (см. серьёзные #2).
- **Asyncio только** — нет celery/APScheduler. Telegram bot polling, JobQueue cleanup (5 минут), `_auto_demo` в `lifespan`, WebSocket `/ws`. Это легковесно и хорошо подходит для shared VPS.
- **Нет PyTorch / JAX / transformers** — единый ML-стек на TensorFlow. Не плодим дубли.
- **Bind mount `.:/app` в `docker-compose.yml`** — код берётся прямо из директории, `git pull` обновляет без пересборки. Удобно для production VPS, но НЕ работает для immutable image-based deploy через registry (если когда-то решим перейти).

## Open questions для будущих сессий

1. ~~PDF-генерация: texlive нужен?~~ → **Да**, оставляем (решено 2026-04-07)
2. Drone bot — отдельный полноценный продукт или вспомогательный сервис? От ответа зависит подход к унификации.
3. ~~ФГИС-ЛК: стаб удалить?~~ → **Нет** — активно используется: 3 REST эндпоинта, RAG enrichment, тесты (решено 2026-04-07)
4. Yandex Workflows: есть ли смысл достраивать pipeline сейчас, или это `wishlist` для версии после грантов?

## Дополнительные находки (2026-04-07)

- **1 pre-existing баг**: `tests/test_classify_api.py::test_classify_via_edge_handles_connection_error` — мокает весь `httpx` модуль через MagicMock, `except httpx.ConnectError` падает. Фикс: мокать `httpx.post`, не весь модуль.
- **`test_drone_bot.py` и `test_bot_workflow.py`** заменяют `sys.modules["cloud.interface.main"]` на MagicMock на уровне модуля — ломает тесты, которые импортируют `app` после них. Тесты route_registry обходят это через `_get_real_app()`.
- **Demo pipeline** (`_run_demo`) — 250+ LOC в main.py, сильно переплетён с `broadcast()`. Следующий кандидат на вынос после bot_handlers split.
