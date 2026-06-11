# Migration inventory (DEFERRED) — 2026-05-01

**Статус:** отложено по решению пользователя 2026-05-01. Тикеты создаются вручную через `/ticket-create` по мере поступления.

Источник: 3 параллельных Explore-сабагента над faun-codebase + docs. Инструкция всем: цитата `file:line`, не дублировать `tasks/audit-roadmap.md` (10 живых пунктов оттуда переданы в exclusion-list).

**Сырых items: 119** (A:22 + B:25 + C:72). После dedup ≈100–110 уникальных (пересечения A↔C: `_run_demo` import_error, hardcoded localhost:8000 в incident.py, librosa drop, retry YandexGPT timeout).

---

## Agent A — Code TODO/FIXME (22 items)

Scope: cloud/, edge/, gateway/, simulator/, devices/, tests/

### TODO core (2)
- [TODO] Yandex Workflows API stub — implement real API when available | `cloud/workflows/yandex_workflows.py:31` | category=research | Lines 32-38 закомментированы, stub возвращает mock в local mode
- [TODO] Retrieve actual photo from companion computer camera | `edge/drone/ardupilot.py:133` | category=coding | Lines 134-139 описывают 4 опции (picamera2, GoPro WiFi, MAVLink, shared FS), сейчас placeholder JPEG bytes

### Silent excepts без логирования (3)
- [BUG] Silent except в `rag_agent.get_forest_unit` | `cloud/agent/rag_agent.py:309` | category=bug | permit_status дефолтится "не проверялось", без log/re-raise
- [BUG] Silent except в `rag_agent.has_valid_permit` | `cloud/agent/rag_agent.py:316` | category=bug | то же — нет log/raise
- [BUG] Silent except в `_import_demo_deps` | `cloud/interface/main.py:400` | category=bug | broadcast `import_error` без имени модуля; пропустил scipy-регрессию 2026-04-09

### Hardcoded values (4)
- [INFRA] Hardcoded `localhost:8000` в incident handler | `cloud/notify/handlers/incident.py:320` | category=infra | dispatch-drone URL — должен быть env CLOUD_API_URL (см. `gateway/relay.py:25`)
- [INFRA] Hardcoded duckdns.org URL в menu button | `cloud/notify/bot_app.py:46` | category=infra | "https://faun-forrest.duckdns.org/" — нужен env BOT_MENU_URL
- [INFRA] Hardcoded TF Hub URLs (2 места) | `edge/audio/classifier.py:76,91` | category=unclear | Если tfhub.dev обновит URL — код сломается; env YAMNET_HUB_URL и YAMNET_CLASS_MAP_URL
- [INFRA] /tmp paths без TempDir context | `cloud/interface/routers/classification.py:83` | category=infra | f"/tmp/live_{uuid.uuid4()}.webm" — лучше `tempfile.TemporaryDirectory()`

### Magic numbers (2)
- [INFRA] `m_per_deg_lat = 111_320.0` без коммента | `cloud/db/microphones.py:217` | category=infra | meters per degree lat — задокументировать "WGS84 Earth circumference / 360°"
- [INFRA] `spacing_m = 350.0` без env-override | `cloud/db/microphones.py:201` | category=infra | env MIC_GRID_SPACING_M

### Dead code (2)
- [CODING] Закомментированный async POST в Yandex Workflows | `cloud/workflows/yandex_workflows.py:32-38` | category=coding | 8 строк commented — реализовать или удалить
- [CODING] Закомментированные опции photo source | `edge/drone/ardupilot.py:134-140` | category=coding | 7 строк options — в task tracker, не в коде

### Bugs / refactors
- [BUG] Все exceptions в classify_api.py → "unknown" | `edge/classify_api.py:44` | category=bug | Нельзя дебажить из API response — distinguish IO/model/format errors
- [CODING] PROXIMITY_RADIUS_M=1000 как local var | `cloud/notify/handlers/incident.py:125` | category=coding | должна быть module-level const

### Infra
- [INFRA] `print()` вместо logger в `gateway/relay.py` | строки 43, 87, 89, 119, 131, 173, 189, 199, 201 | category=infra | 9 мест, production code — logger
- [INFRA] Нет retry на YandexGPT timeout | `cloud/agent/decision.py:111` | category=infra | single attempt, нужен backoff (3 попытки 2s/5s/10s)
- [INFRA] Generic except в datasphere_client | `cloud/agent/datasphere_client.py:49` | category=infra | без parsing error codes, без exponential backoff на rate limit

---

## Agent B — Planning docs / notebooks (25 items)

Scope: tasks/*.md (минус audit-roadmap), docs/, docs/notebooks/*.ipynb, README.md, CLAUDE.md

### DataLens / dashboards (3)
- [DataLens дашборд: починка 2 сломанных виджетов] | `tasks/datalens-fix-prompt.md` (всё) | category=content | ERR.CHARTS.DATA_FETCHING_ERROR, ERR.UNKNOWN, переименование виджетов
- [YDB Serverless setup полный] | `tasks/datalens-ydb-setup.md` (всё) | category=infra | Создать YDB БД (faun-incidents), SA с ydb.editor, подключить DataLens к API
- [DataLens интеграция API Connector] | `docs/deployment.md`:YDB Setup | category=infra | URL: http://81.85.73.178:8000/api/v1/datalens/incidents — переключить чарты на реальные данные

### ML research (7)
- [YAMNet v8 fine-tuning на реальных датасетах] | `tasks/prompt_finetune_v8.md` (всё) | category=research | ESC-50/UrbanSound8K/FSC22, проверка на demo
- [FPR-тест background-only миксов] | `docs/notebooks/02_yamnet_test.ipynb`: TODO-02 (5×) | category=research | False positive rate на 300 чистых фоновых миксах
- [Confidence gating: пороги и validation] | `docs/notebooks/02_yamnet_test.ipynb`: TODO-16 (7×) | category=research | 3-уровневая система (log<0.4 / verify 0.4-0.7 / alert>0.7)
- [WIP: YAMNet test notebook 02] | `docs/notebooks/02_yamnet_test.ipynb`: WIP (5×) | category=research | Несколько частей помечены WIP — статус неясен
- [WIP: Distance estimation notebook 03] | `docs/notebooks/03_distance_estimation.ipynb`: WIP | category=research
- [Ноутбук 01: chain saw class identification] | `docs/notebooks/01_data_and_mix.ipynb:sec3e:~230` | category=research | Placeholder CHAINSAW_CLASS_ID = "???" в FSC22
- [RAG fallback chain completeness] | `docs/yandex-cloud.md:RAG#2-4` | category=research | SDK → plain YandexGPT → static, нет покрытия partial failures (File Search OK + Web Search падает)

### REFINES к audit-roadmap (5)
- [REFINES medium#3] Унифицировать polling ranger_bot/drone_bot | `tasks/audit-roadmap.md`:средние#3 | category=coding
- [REFINES open#2] Drone Bot — продукт vs сервис | `tasks/audit-roadmap.md`:open#2 | category=unclear | решение влияет на унификацию
- [REFINES доп.#3] Demo pipeline вынести из main.py | `tasks/audit-roadmap.md`:доп.#3 | category=coding | 250+ LOC, переплетён с broadcast()
- [REFINES доп.#4] `_run_demo` обобщённый import_error | `tasks/audit-roadmap.md`:доп.#4 | category=coding
- [REFINES доп.#5] Drop librosa из edge image | `tasks/audit-roadmap.md`:доп.#5 | category=infra | -300 MB

### Integration / coding (4)
- [Yandex Workflows достраивать?] | `CLAUDE.md`:open#4 | category=unclear | wishlist для версии после грантов?
- [Vision classifier prompt-update в deployment.md] | `docs/yandex-cloud.md`:Vision (всё) | category=coding | VisionResult доработана (equipment_types, people_count, damage_area), не упоминается в deployment
- [ФГИС-ЛК endpoint заглушка → real] | `docs/api.md`:FGIS-LK (всё) | category=coding | 3 эндпоинта stub: forest-unit, permits, violation
- [Protocol PDF кэширование и улучшения] | `docs/api.md`:Protocol PDF (24-30) | category=coding | PDF в `incident.protocol_pdf`, fpdf2 fallback — нет учёта v2

### Lessons follow-ups (3)
- [Lesson #9: TF OOM, нужен DISABLE_AUTO_DEMO=1 в healthcheck timeout 360s] | `tasks/lessons.md:L9` | category=infra
- [Lesson #10: тяжёлый python healthcheck (~40MB) vs curl (~5MB)] | `tasks/lessons.md:L10` | category=infra | уже учтено в deployment.md
- [Lesson #11: split deps → ImportError в ленивых импортах] | `tasks/lessons.md:L11` | category=coding | scipy скрылся в `_import_demo_deps()` после split

### Infra (3)
- [Yandex Cloud API timeout margin] | `docs/yandex-cloud.md:RAG#117` | category=infra | RAG_SDK_TIMEOUT=15s, нет обсуждения network latency
- [Deployment healthcheck SLA] | `docs/deployment.md:17-18` | category=infra | interval 120s, timeout 10s, start_period 360s — оптимально?
- [CloudSQL/YDB migration path] | `CLAUDE.md`:Audit roadmap#2 | category=infra | заблокирована отсутствием local-YDB / эмулятора
- [Test coverage DataLens endpoints] | `docs/testing.md:60` | category=coding | нет integration/E2E тестов для datalens endpoints

---

## Agent C — Infra / CI / Tests / Security (72 items)

Scope: .github/workflows, Dockerfile*, requirements*, tests/conftest+skip-xfail, .gitignore, nginx/, docs/deployment.md, tasks/datalens-* + lessons

### [BLOCKER] (1)
- Sync `pool.retry_operation_sync()` в async Telegram-handler | `cloud/notify/handlers/alerts.py` (или incident.py) | category=bug | lessons #2 предупреждал. Блокирует event loop под YDB rate-limit. Нужен `asyncio.to_thread()` wrapper.

### Critical [SEC] (13)
- [SEC] No-auth REST API: DELETE/PATCH/POST на rangers/permits/mics | `cloud/interface/routers/rangers.py:33-77` | category=bug | критично, любой удалит ranger/permit
- [SEC] No rate limiting на admin endpoints | `cloud/interface/routers/rangers.py:33` | category=bug | + no auth = brute-force/DoS
- [SEC] Containers run as root (4 Dockerfile) | `Dockerfile`, `cloud/Dockerfile`, `edge/Dockerfile`, `gateway/Dockerfile` | category=security | groupadd -r appuser
- [SEC] base image `python:3.11-slim` без patch-pin | все Dockerfile:1 | category=security | latest patch — pin to `python:3.11.10-slim`
- [SEC] YDB SA key file world-readable в bind mount | `cloud/db/ydb_client.py:143` | category=security | chmod 400 / secrets manager
- [SEC] PLANE_API_KEY в .env (риск ротации) | `.env:1` | category=security | если был committed когда-то — rotate; gh history check
- [SEC] No CORS middleware | `cloud/interface/main.py` | category=bug | frontend→cloud API XHR работает только через nginx proxy
- [SEC] No file size/MIME check на upload | `cloud/interface/routers/classification.py` | category=security | OOM via large upload
- [SEC] WebSocket /ws без message rate limit | `cloud/interface/main.py:websocket_endpoint` | category=security | flood-DoS
- [SEC] Lat/lon без bounds check | `cloud/interface/routers/rangers.py:RangerCreate` | category=security | (-1000, 1000) ломает TDOA
- [SEC] No HTTPS redirect / HSTS | `cloud/interface/main.py` | category=security | port 8000 plaintext если bypass nginx
- [SEC] No rate limiting на RAG endpoint | `cloud/interface/routers/rag.py` | category=security | дорогой Yandex AI call → DoS
- [SEC] EXIF не стрипается на drone-фото в protocol PDF | `cloud/agent/protocol_pdf.py` | category=security | leak GPS координат

### Bugs (18)
- Demo API hardcoded localhost:8000 | `cloud/notify/handlers/incident.py:1` | category=bug | работает только локально
- Edge classify retry blocking time.sleep | `cloud/interface/main.py:56-58` | category=bug | `_classify_via_edge()` 2s sleep — нужен `asyncio.sleep`
- Sync YDB ops в async без to_thread | `cloud/db/ydb_rangers.py, ydb_microphones.py, ydb_permits.py` | category=bug
- No transaction rollback в multi-table inserts | `cloud/db/ydb_incidents.py` | category=bug | YDB auto-rollback но не documented
- YDB rate-limit errors не retry в части paths | lessons #3 | category=bug
- Demo pipeline imports scipy после split deps | `cloud/interface/main.py:400` | category=bug | NameError на demo click
- Module-level sync seed_microphones risk | lessons #6 | category=bug | sub-zero (mitigated в lifespan)
- Yandex Workflows stub без env-gate | `cloud/workflows/yandex_workflows.py:8` | category=bug | TODO в production
- Auto-demo flood risk | `cloud/interface/main.py:_run_demo` | category=bug | back-to-back deploys flood
- No rate limiting на Telegram /incident | `cloud/notify/handlers/incident.py` | category=security | spam 100s incidents
- WebSocket broadcasts ВСЕМ клиентам | `cloud/interface/main.py:broadcast()` | category=security | filter by zone/ranger
- Class taxonomy desync edge ↔ cloud | `edge/audio/classifier.py` vs `cloud/decision/decider.py` | category=bug
- Permission на /app/sa-key.json не enforced | `cloud/db/ydb_client.py:143` | category=security
- test_drone_bot.py replaces sys.modules at module level | `tests/test_drone_bot.py` | category=bug | lessons #8 fragile
- Conftest cleanup _last_sent state не auto в crashed tests | `tests/conftest.py` | category=bug
- Decider thresholds hardcoded | `edge/decision/decider.py` | category=unclear | env override для tuning
- YAMNet model (6.4 MB) не versioned | `edge/audio/classifier.py` | category=infra | s3/HF
- Hardcoded demo audio URLs | `demo/generate_audio.py` | category=infra

### CI / Build (10)
- CI cache missing на test runs | `.github/workflows/ci.yml:18` | category=infra | actions/cache@v4 для requirements
- CI workflow branches включают Oleg | `.github/workflows/ci.yml:5` | category=unclear | `[main]` only
- CI installs pytest ad-hoc | `.github/workflows/ci.yml:17` | category=infra | requirements-test.txt
- No code coverage в CI | `.github/workflows/ci.yml` | category=infra | pytest-cov + threshold + badge
- Requirements upper bounds не tested | `requirements.txt` | category=infra | boundary version test
- No Python matrix в CI | `.github/workflows/ci.yml` | category=infra | 3.12 / 3.13
- Ruff cache в .gitignore — uncached CI lint | `.gitignore:5` | category=infra | minor
- No staged rollout / canary | `.github/workflows/deploy.yml:28` | category=infra | all-or-nothing
- No rollback в deploy.yml | `.github/workflows/deploy.yml:24-32` | category=infra | tag images by SHA
- No secrets rotation reminder | `.github/workflows/deploy.yml` | category=infra | quarterly issue/calendar

### Deploy / runtime (8)
- Healthcheck start_period 360s vs TF 6 min | `docker-compose.yml:26` | category=infra | bump to 480s
- SQLite DB files не в named volume | `docker-compose.yml:54-55` | category=infra | бекап стратегия
- No backup стратегии для DB / model | `docs/deployment.md` | category=infra
- No graceful shutdown в edge server | `edge/server.py` | category=infra | SIGTERM handler
- Microphones seed на каждом startup | `cloud/db/microphones.py + main.py lifespan` | category=infra | INSERT OR IGNORE / count check
- No liveness probe для edge | `edge/server.py` | category=infra | /health endpoint
- Telegram token не rotated | `docs/deployment.md:84` | category=infra | rotation schedule
- FGIS-ЛК API key не rotated | `cloud/workflows/yandex_workflows.py` | category=infra

### Observability (7)
- No centralized logging sink | `cloud/` | category=infra | Sentry/Datadog/JSON struct logs
- No alerting on healthcheck failures | `docker-compose.yml:22-26` | category=infra | docker-compose не auto-restart на red
- No metrics export (Prom/StatsD) | `cloud/` | category=infra
- YDB SDK timeout global per-op | `docs/deployment.md:119` | category=infra | per-endpoint timeouts
- Logs без request ID | `cloud/` | category=infra | correlation ID для tracing
- No monitoring LoRa gateway perf | `gateway/relay.py` | category=infra | metrics throughput/errors
- No auto-cleanup старых incidents | `cloud/db/incidents.py` | category=infra | TTL / archival

### Tests (5)
- Tests мокают Telegram bot completely | `tests/test_bot_workflow.py` | category=infra | monthly live smoke
- Test isolation не enforced между классами | `tests/test_bot_edge_cases.py` | category=infra | teardown_method
- No integration test cloud↔edge HTTP | `tests/` | category=infra | contract testing (Pact / JSON schema)
- No backend coverage SQLite vs YDB | `tests/conftest.py` | category=unclear | dual-backend desync
- Incident PDF size limit | `cloud/agent/protocol_pdf.py` | category=security | OOM via large incident

### Misc unclear (6)
- Demo scenarios hardcoded ["chainsaw","gunshot","engine"] | `cloud/interface/main.py:75` | category=unclear | sync с decider params
- Quiet hours TZ DST | `cloud/notify/telegram.py:QUIET_HOURS_START=22` | category=unclear | pytz.timezone("Europe/Moscow")
- API versioning beyond /api/v1 | `cloud/interface/routers/*.py` | category=unclear | deprecation headers
- Gateway relay deduplication | `gateway/relay.py` | category=unclear | message ID dedup
- Timezone в incident timestamps implicit | `cloud/db/incidents.py` | category=unclear | UTC explicit
- No Content-Type validation on JSON endpoints | `cloud/interface/routers/rag.py, classification.py` | category=security

---

## Stats

| Category | Count |
|---|---|
| bug | ~25 |
| security | 13 |
| infra | ~50 |
| coding | ~12 |
| research | ~12 |
| content | 1 |
| unclear | ~8 |
| **TOTAL (раньше dedup)** | **119** |
| **[BLOCKER]** | 1 |
| **[SEC]** | 13 |

Резервный файл — на случай если решишь вернуться к bulk-миграции.
