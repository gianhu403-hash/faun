# Faun — офлайн batch-pipeline распознавания видов птиц

## Контекст

Проект развернулся из хакатонного прототипа акустического мониторинга леса в
**офлайн batch-pipeline распознавания ВИДОВ ПТИЦ** по записям с аудиоловушек.
Заказчик/партнёры: Yandex Cloud + Президентский фонд природы. Пилот стартует с июля.

**Продукт:** оператор загружает архив записей с ловушек (папка/URL) → система
детектирует звуковые события, классифицирует виды птиц и отдаёт CSV с треками,
таймкодами, видом и вероятностью. Плюс простой web-UI для запуска и скачивания.

## Архитектура (v2 pipeline)

Двухэтапный контур: **детектор событий → классификатор видов**.

Пакет `faun/`:
- `faun/ingest` — скан папок ловушек + `info.txt` → Manifest (одна папка = одна ловушка A1..A5).
- `faun/ordering` — хронологическое упорядочивание записей манифеста.
- `faun/segmentation` — `SegmentExtractor`: downmix mono + resample 48k→16k → onset-детектор (`faun/ml/onset.py`).
- `faun/classification` — протокол `SpeciesClassifier` + адаптеры:
  - `StubAdapter` (в скелете), `BirdNETAdapter`, `PerchAdapter`,
  - `YAMNetAdapter` (embeddings + probe, НЕ хакатонная YAMNet-голова).
- `faun/output` — `CsvWriter`, колонки CSV: `track,start_sec,duration_sec,species,probability` (+ sidecar с метаданными ловушки).
- `faun/jobs` — изоляция батчей: namespace на `job_id`, `workdir=jobs_root/<job_id>/`, без общих temp-путей.
- `faun/storage` — протокол `Storage` + только `LocalFSStorage` (S3 — задача на июль, НЕ сейчас).
- `faun/api` — FastAPI: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/results.csv`.
- `faun/cli` — `faun process <dir> [--out results.csv]`.
- `faun/static/index.html` — vanilla-JS UI: форма (папка/URL) → `POST /jobs` → poll статуса → таблица + скачивание CSV.
- `faun/ml` — REUSE-ядро из хакатона: `onset.py`, `ndsi.py`, `yamnet.py`, `datasphere_client.py`.

Контракт интерфейсов заморожен в `faun/INTERFACES.md` — Phase-2 волны пишут против него, сигнатуры не меняем.

## Вычислитель

Кластер **cluster-alex** (RTX 2060S 8GB). TF/JAX в CI и локально — **CPU-only**
(GPU только на кластере). Веса `faun/ml/yamnet_forest_classifier_v7.keras`
(~6.4 MB) в `.gitignore` — раскладываются на кластер вручную, TF импортируется
лениво внутри функций `faun/ml/yamnet.py`.

## Legacy

Весь хакатонный код заморожен в `legacy/` и на теге **v1-hackathon**:
real-time мониторинг, YAMNet-голова, TDOA-триангуляция, LoRa-mesh, Telegram-боты,
дрон (ArduPilot), RAG-агент, Yandex Workflows. Импорты внутри `legacy/` НЕ чиним.
Демо хакатона продолжает крутиться на кластере. Тесты legacy лежат в
`legacy/tests/` и CI их НЕ гоняет (CI запускает только `tests/`).

Полезное из legacy (как справка): `FOLDER_ID=b1g5lqh1mqg84cabtejb`,
`SEARCH_INDEX_ID=fvttk7bjvnm39qogtoep` — относятся к legacy RAG, в v2 не используются.

## Инфраструктура

- Ingress: `faun.antopkin.ru` → nginx `anchor` → кластер cluster-alex.
- **Forgejo на кластере НЕ трогать** — работаем только с GitHub.
- Ветка для коммитов: `main`. CI: `.github/workflows/ci.yml` (deploy.yml — заглушка).

## Инструкции

- Язык общения: русский.
- Модели (`*.keras`, `*.h5`, `*.npz`) в `.gitignore`.
- Зависимости pipeline: `requirements-pipeline.txt`. Legacy-requirements — в `legacy/`.
