# Faun — Code Audit Roadmap

Зафиксировано 2026-04-07 в ходе подготовки к деплою на shared VPS `delphi-press`. Цель документа — собрать находки аудита в одном месте, чтобы в будущих сессиях по ним можно было идти точечно. Приоритет «делать сейчас» отсутствует — это roadmap, а не todo.

## Базовые метрики проекта (snapshot)

- **Python код**: 110 файлов, 21 564 строки
- **cloud/**: 8 286 строк (`db/` 32 %, `notify/` 25 %, `interface/` 16 %, `agent/` 15 %)
- **edge/**: 1 748 строк + ML модель `yamnet_forest_classifier_v7.keras` (6.4 MB)
- **gateway/**: 348 строк
- **tests/**: 8 418 строк, 31 файл, ~417 тестов
- **Финальный docker-образ**: ~1.8–2.0 GB (`python:3.11-slim` + `texlive-*` + `requirements.txt` ≈ 793 MB установленных пакетов)
- **RAM при старте**: cloud ≈ 440 MB, edge ≈ 955 MB (TF + YAMNet), lora_gateway ≈ 50 MB → суммарно ≈ 1.4 GB

## Quick wins (низкая стоимость, заметный эффект)

| # | Изменение | Эффект | Где |
|---|---|---|---|
| 1 | Убрать `texlive-*` из `Dockerfile` или вынести в multi-stage | −300 MB образа | `Dockerfile` |
| 2 | Multi-stage build (builder → runtime) | −500 MB финального образа | `Dockerfile` |
| 3 | Pin upper bounds зависимостей: `tensorflow<2.16`, `numpy<2.0`, `scipy<1.13` | Стабильность production | `requirements.txt` |
| 4 | Отдельные `Dockerfile` для cloud / edge / gateway. У gateway TF не нужен | Gateway образ −500 MB | `cloud/Dockerfile`, `edge/Dockerfile`, `gateway/Dockerfile` + `docker-compose.yml` |
| 5 | Удалить мёртвый стаб `cloud/integrations/fgis_lk.py` (169 строк, нет публичного API ФГИС-ЛК) | −169 строк | `cloud/integrations/fgis_lk.py`, импорты |
| 6 | Удалить заглушку `cloud/workflows/yandex_workflows.py` (61 строка, всегда возвращает пустой результат) | −61 строка | `cloud/workflows/yandex_workflows.py` |
| 7 | Закрыть TODO в `cloud/agent/decision.py:11` (Yandex Foundation Models completions) и `cloud/interface/main.py:214` (REST для frontend) — либо реализовать, либо удалить | Чистота кода | указанные файлы |

## Средние улучшения (требуют рефакторинга, но без архитектурных рисков)

| # | Изменение | Что даёт | Где |
|---|---|---|---|
| 1 | Split монолита `cloud/interface/main.py` (1 307 строк) на FastAPI routers по доменам: `routers/health.py`, `routers/incidents.py`, `routers/demo.py`, `routers/photos.py`, `routers/analytics.py`, `routers/ws.py` | Проще навигация и тесты | `cloud/interface/main.py` |
| 2 | Split `cloud/notify/bot_handlers.py` (1 038 строк) на handlers по доменам: `handlers/registration.py`, `handlers/incidents.py`, `handlers/photos.py`, `handlers/voice.py` | То же | `cloud/notify/bot_handlers.py` |
| 3 | Унифицировать `ranger_bot` и `drone_bot` (общая инфраструктура polling/JobQueue, разные handlers/команды) | Меньше дублирования в `cloud/notify/` | `cloud/notify/bot_app.py`, `drone_bot_app.py` |
| 4 | Покрыть тестами split-части (минимум один test-файл на router/handler) | Не уронить функциональность при split | `tests/cloud/...` |

## Серьёзные изменения (под отдельный пилот, нужно дополнительное решение)

| # | Изменение | Эффект | Риски |
|---|---|---|---|
| 1 | TensorFlow → TFLite в edge | edge образ ~−500 MB, RAM ~−400 MB | Конвертация модели + регрессионные тесты на классификацию + переучивание тестов TDOA |
| 2 | Перевести cloud на YDB-only, удалить SQLite-impl (~1 300 строк в `cloud/db/`) | −1 300 строк, −12 файлов, единый DB-бэкенд | Локальная разработка усложнится — нужен local-YDB или YDB-эмулятор |
| 3 | Заменить `python-telegram-bot[job-queue]` на легковесный aiogram + внешний планировщик | Меньше зависимостей в cloud | Полностью переписать handlers и polling-лайфсайкл |
| 4 | Вынести модель YAMNet в Hugging Face Releases / Yandex Object Storage и качать на старте контейнера | Не таскать `*.keras` через scp | Требует CDN/S3 и cache invalidation |

## Архитектурные наблюдения (для контекста)

- **Дублирование SQLite/YDB в `cloud/db/`** оправдано factory pattern (`db/factory.py`): локально SQLite, в production YDB. Удалять только вместе с переходом на YDB-only (см. серьёзные #2).
- **Asyncio только** — нет celery/APScheduler. Telegram bot polling, JobQueue cleanup (5 минут), `_auto_demo` в `lifespan`, WebSocket `/ws`. Это легковесно и хорошо подходит для shared VPS.
- **Нет PyTorch / JAX / transformers** — единый ML-стек на TensorFlow. Не плодим дубли.
- **Bind mount `.:/app` в `docker-compose.yml`** — код берётся прямо из директории, `git pull` обновляет без пересборки. Удобно для production VPS, но НЕ работает для immutable image-based deploy через registry (если когда-то решим перейти).

## Open questions для будущих сессий

1. PDF-генерация: нужен ли реально `texlive` в production, или можно перейти на `weasyprint` / `reportlab` (значительно меньше)?
2. Drone bot — отдельный полноценный продукт или вспомогательный сервис? От ответа зависит подход к унификации.
3. ФГИС-ЛК: появился ли публичный API за время развития? Если нет — стаб удаляем.
4. Yandex Workflows: есть ли смысл достраивать pipeline сейчас, или это `wishlist` для версии после грантов?
