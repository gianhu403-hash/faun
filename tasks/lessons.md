# Lessons Learned

## 1. Vision stub не должен быть "безопасным"

False negative хуже false positive для систем безопасности. Если Vision API
недоступен, stub должен сигнализировать потенциальную угрозу (`has_felling=True`),
чтобы pipeline продолжил работу и инспектор мог проверить вручную.

## 2. sync YDB из async = блокировка event loop

`pool.retry_operation_sync()` — синхронный вызов. Вызов из async-хандлера Telegram
блокирует event loop. При YDB rate limiting (`ResourceExhausted`) ретраи с
`time.sleep()` полностью замораживают обработку сообщений.

**Правило:** всегда `await asyncio.to_thread(create_incident, ...)` для YDB
операций из async-контекста.

## 3. YDB Serverless rate limits

Бесплатный тарифный план YDB Serverless имеет жёсткие RU-лимиты. Bulk upserts
15 batch по 200 строк исчерпывают квоту начиная с Batch 2.

**Правила:**
- `batch_size = 500` (меньше gRPC вызовов)
- `time.sleep(1.0)` между батчами (не 0.5)
- `wait = 2 ** (attempt + 1)` для retry (начинать с 2s, не 1s)
- DDL операции тоже throttle: `time.sleep(0.5)` между ними

## 4. Тестировать деградацию

Если Vision API упал, pipeline должен продолжать работать. Нельзя полагаться на
внешние API без graceful degradation. JSON от Vision может быть битым — `json.loads`
обязательно в `try/except`.

## 5. `session.prepare()` vs typed tuples

`session.prepare()` парсит DECLARE и сам типизирует параметры. Typed tuples
(`TypedValue(PrimitiveType.Utf8, value)`) нужны ТОЛЬКО без `prepare()`.
Не надо дублировать типизацию — это вызывает `double type binding` ошибки.

## 6. Module-level sync код блокирует import

`seed_microphones()` на уровне модуля (вне `lifespan`) вызывается синхронно при
`import main`. Если YDB недоступен или медленный — весь FastAPI зависает на старте.

**Правило:** тяжёлые sync-операции перенести в `lifespan` через `asyncio.to_thread()`.

## 7. Docker volume mount + restart ≠ свежий код

При `docker compose restart` контейнер НЕ пересоздаётся — может закэшировать старый
`.pyc` или состояние модулей. Для деплоя с volume mount:
- `docker compose up -d --force-recreate` (пересоздать контейнер)
- Или `docker compose up --build -d` (пересобрать образ + пересоздать)
- Добавить `ENV PYTHONDONTWRITEBYTECODE=1` в Dockerfile чтобы Python не писал `.pyc`

## 8. При рефакторинге — проверять все использования удалённых переменных

Удалили `mic_coords = [...]` при рефакторинге `_run_demo()`, но `MicSimulator`
использовал `mic_coords` ниже по коду. `grep mic_coords` перед удалением спас бы
от NameError на проде.

## 9. OOM Kill от дублирования TF в контейнерах

Cloud-контейнер импортирует `from edge.audio.classifier import classify` напрямую
(Python import, не HTTP). Это загружает TensorFlow (~500-800 MB) повторно —
edge уже держит свой экземпляр TF. На VPS с 1.9 GB RAM два TF = OOM Kill
(exitCode=137, SIGKILL).

**Правила:**
- Auto-demo (`_auto_demo()`) триггерит TF-загрузку; на малой RAM → `DISABLE_AUTO_DEMO=1`
- Healthcheck: `curl -sf` вместо `python -c "..."` (экономия ~40 MB на проверку)
- `start_period: 360s` для healthcheck (TF грузится ~6 минут)
- Долгосрочно: cloud → edge по HTTP, не через Python import

## 10. Healthcheck не должен быть тяжелее самого сервиса

`python -c "import urllib.request; ..."` запускает полный Python-интерпретатор
(~40 MB) для каждой проверки. На VPS с ограниченной RAM это усиливает memory
pressure во время загрузки TF. `curl -sf url` использует ~5 MB и работает мгновенно.

## 11. Split requirements → латентные ImportError в ленивых импортах

После split `requirements.txt` на `requirements-cloud.txt` / `requirements-edge.txt`
(коммиты `20839d9` + `3db3adf`, 2026-04-07) из cloud-контейнера ушли `tensorflow`,
`scipy`, `librosa`. Классификация была переведена на HTTP через `_classify_via_edge()`,
но `cloud/interface/main.py::_import_demo_deps()` продолжал напрямую импортировать
`edge.tdoa.triangulate` — а он тянет `scipy` на module-level. Контейнер стартовал
нормально (верхние импорты `main.py` scipy не трогают), но при клике «Бензопила» в
дашборде ленивый import падал → `except Exception` → фронту уходило `reason: "import_error"`.

Баг жил ~2 суток незамеченным: auto-demo отключён на VPS через `DISABLE_AUTO_DEMO=1`
(см. урок №9), а тесты мокают `_import_demo_deps()` целиком, поэтому реальная dep-цепочка
не проверялась.

**Правила:**
- При split deps делать `grep -r "from edge\." cloud/` и проверять каждый импорт на
  транзитивные тяжёлые зависимости (scipy, tensorflow, librosa, sounddevice).
- Для ленивых импортов, которые ловятся через `except Exception`, — логировать **имя
  конкретного модуля**, который упал, а не просто `"import_error"`. Сейчас это скрыто
  в `logger.exception`, но наружу в WebSocket уходит только generic строка.
- Rule of thumb: если cloud больше не содержит TF, он и не должен содержать модули,
  чей транзитив тянет TF/scipy. Либо переносить в edge+HTTP (как classifier), либо
  добавлять dep обратно с осознанной причиной (triangulate → scipy — гео-вычисление
  имеет смысл на cloud-стороне для симуляции, оставляем scipy).
