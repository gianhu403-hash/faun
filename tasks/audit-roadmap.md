# Faun — Code Audit Roadmap

Зафиксировано 2026-04-07 в ходе подготовки к деплою на shared VPS `delphi-press`. Обновлено 2026-04-07 после split main.py. Цель документа — собрать находки аудита в одном месте, чтобы в будущих сессиях по ним можно было идти точечно.

## Базовые метрики проекта (обновлено 2026-04-07 после bot_handlers split)

- **Python код**: ~125 файлов, ~22 800 строк
- **cloud/**: `interface/main.py` 639 строк (было 1 307), 8 routers, `notify/handlers/` 7 модулей (было `bot_handlers.py` 1 038 LOC → 50 LOC shim)
- **edge/**: 1 748 строк + ML модель `yamnet_forest_classifier_v7.keras` (6.4 MB)
- **gateway/**: 348 строк
- **tests/**: 514 тестов (было 453), CI ✅ зелёный, локально 514/514
- **Docker images**: cloud 1.38 GB (было 2.3), edge 3.58 GB, gateway 0.29 GB (было 2.3) — total 5.25 GB (было 6.9 GB)
- **VPS диск**: 59% (было 89%), auto-cleanup в deploy.yml

## Quick wins

| # | Изменение | Эффект | Статус |
|---|---|---|---|
| 1 | ~~Убрать `texlive-*`~~ | — | **Оставляем** — нужен для PDF-протоколов |
| 2 | ~~Multi-stage build~~ | — | **Поглощено #4** — per-service Dockerfiles дают больше |
| 3 | Pin upper bounds зависимостей | Стабильность production | **Сделано** — `requirements.txt` с upper bounds + `requirements.lock` |
| 4 | Отдельные `Dockerfile` для cloud/edge/gateway | Gateway −88%, cloud −40% | **Сделано** — total −24% (6.9 GB → 5.25 GB) |
| 5 | ~~Удалить `fgis_lk.py`~~ | — | **Отменено** — активно используется |
| 6 | ~~Удалить `yandex_workflows.py`~~ | — | **Отменено** — в критическом пути |
| 7 | Закрыть устаревшие TODO | Чистота кода | **Сделано** — 2 TODO удалены, 2 оставлены (production blockers) |

## Средние улучшения

| # | Изменение | Что даёт | Статус |
|---|---|---|---|
| 1 | Split `cloud/interface/main.py` | Проще навигация и тесты | **Сделано** — 1 307 → 639 LOC, 8 routers |
| 2 | Split `cloud/notify/bot_handlers.py` | То же | **Сделано** — 1 038 → 50 LOC shim, 7 modules в `handlers/` |
| 3 | Унифицировать `ranger_bot` и `drone_bot` | Меньше дублирования | **TODO** — дублирующая инфраструктура polling |
| 4 | Покрыть тестами split-части | Safety net | **Сделано** — 45 characterization tests |
| 5 | Fix flaky CI tests (rate limit bug) | CI зелёный | **Сделано** — production баг в `_is_rate_limited` + fixture cleanup |

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

- ~~`test_classify_api.py::test_classify_via_edge_handles_connection_error`~~ → **Сделано** — патчим `httpx.post`, не весь модуль
- **`test_drone_bot.py` и `test_bot_workflow.py`** заменяют `sys.modules["cloud.interface.main"]` на MagicMock на уровне модуля — обходим через `_get_real_app()` в новых тестах
- **Demo pipeline** (`_run_demo`) — 250+ LOC в main.py, сильно переплетён с `broadcast()`. Следующий кандидат на вынос
- **Edge image 3.58 GB** — TF 2.21 (`tensorflow` package = `tensorflow-cpu` в 2.16+). Можно сэкономить ~300 MB удалив librosa (используется только для PCEN в v8 модели)
- **Auto-cleanup в deploy.yml** — `docker image prune -f && docker builder prune -f --filter until=24h` после каждого деплоя
- **CSP nginx** — Faun использует отдельный `faun-security-headers.conf` с whitelist для unpkg.com, basemaps.cartocdn.com, datalens.yandex, wss://faun.antopkin.ru
