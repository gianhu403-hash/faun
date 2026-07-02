# Faun — офлайн batch-pipeline распознавания диких животных по звуку

## Контекст

Проект развернулся из хакатонного прототипа акустического мониторинга леса в
**офлайн batch-pipeline распознавания видов животных** по записям с аудиоловушек.
Заказчик/партнёры: Yandex Cloud + Президентский фонд природы. Пилот стартует с июля.

**Целевой объект — скрытные млекопитающие** (например, садовая соня, выхухоль), которых
готовые птичьи классификаторы (BirdNET/Perch) не определяют — это и есть отличие продукта.
**Птицы — уже работающий baseline**: Perch 2 в проде реально их распознаёт (см. Инфраструктуру
ниже), но это не итоговая цель. Размеченного звука целевых млекопитающих сейчас нет нигде — ни
в репозитории, ни на кластере, ни в открытых датасетах (детали и статус — `docs/БРИФИНГ-ML.md`,
`docs/АУДИТ-2026-06-25.md`).

**Продукт:** оператор загружает архив записей с ловушек (папка/URL) → система
детектирует звуковые события, классифицирует виды и отдаёт CSV с треками,
таймкодами, видом и вероятностью. Плюс простой web-UI для запуска и скачивания.

## Архитектура (v2 pipeline)

Двухэтапный контур: **детектор событий → классификатор видов**.

Пакет `faun/`:
- `faun/ingest` — скан папок ловушек + `info.txt` → Manifest (одна папка = одна ловушка A1..A5).
- `faun/ordering` — хронологическое упорядочивание записей манифеста.
- `faun/ordering` — хронологическая сортировка + детект пропусков цикла записи (10 мин + 1 мин паузы).
- `faun/audio` — **единый владелец препроцессинга** (ADR-0002): `downmix`/`resample`(soxr)/`fit_window`. `faun.embeddings` ре-экспортит их как `_downmix/_resample/_fit_window` (frozen-импорт для `training/dataset.py`, инвариант is-идентичности); `segmentation` и `perch_v2._prepare` делегируют (perch_v2 сохраняет свой peak-normalize). Только numpy+soxr.
- `faun/segmentation` — `SegmentExtractor`: downmix mono + resample (`soxr`) 48k→16k → onset-детектор (`faun/ml/onset.py`); препроцессинг делегирован в `faun.audio`.
- `faun/pipeline` — **reusable executor** (ADR-0003): `run_batch` (генератор: read→`SegmentExtractor`→slice→16k→classify→`Detection`) + `slice_clip`/`to_classifier_input`/`CLASSIFY_SR`. `run_pipeline` и `batch_label` — тонкие обёртки над ним; клипы row-aligned с детекциями (контракт для emb-экспорта). TF/torch-free.
- `faun/classification` — протокол `SpeciesClassifier` + адаптеры (ленивый импорт тяжёлых либ):
  - `StubAdapter` (в скелете, без ML), `BirdNETAdapter` (CC BY-NC-SA, non-commercial), `PerchAdapter` (Perch 1, TFHub anon, 1280),
  - `Perch2Adapter` (Perch 2, Apache 2.0, **1536**, kagglehub→`saved_model`; creds-gated → `RuntimeError`, НЕ fallback на Perch 1; `FAUN_CLASSIFIER=perch-v2` / `PERCH_V2_MODEL_PATH`). `_load_labels()` читает `assets/labels.csv` (научные имена, дроп заголовка `inat2024_fsd50k`, 14795 классов); нет assets / рассинхрон длины с логитами → fail-safe `species_<i>` (без off-by-one),
  - `PerchProbeAdapter` (`perch_probe.py`): Perch 2-эмбеддинг(1536) → обученная проба (`PERCH_V2_PROBE_PATH`, pickle) → `Prediction`; нет пробы → `embedding_only`; `FAUN_CLASSIFIER=perch-probe`, source `model:perch-v2-probe`,
  - `YAMNetAdapter` (embeddings + probe, НЕ хакатонная YAMNet-голова).
- `faun/output` — `CsvWriter`, колонки CSV: `track,start_sec,duration_sec,species,probability` (+ sidecar `results_meta.json` с метаданными ловушки).
- `faun/detections` — центральная абстракция **детекции**: `Detection`(переиспользует `Segment`/`Prediction`) + `Label`(`source`/`status`/`ts`); атомарный sidecar `detections.jsonl`; `is_ground_truth` (истина только `expert`/`ranger` + `confirmed`/`corrected`).
- `faun/localization` — порт TDOA из `legacy/edge/tdoa` (GCC-PHAT + Nelder-Mead): `triangulate`, `localize_event` (<3 ловушек → `insufficient-traps` без точки; ≥3 → `tdoa-synthetic-validated`). Валидно только на синтетике; на реале — мс-синхронизация (см. `experiments/report/METRICS_HONESTY.md`).
- `faun/retraining` — петля «человеческие метки → дообучение пробы»: `retrain_from_labels` (фильтр только `expert`/`ranger`, нуль таких → явный отказ), `train_probe_cv` (StratifiedKFold + CI, мультикласс), `save/load_probe` (sklearn pickle, грузится `YAMNetAdapter` через `YAMNET_PROBE_PATH`). Аддитивно: `species_eval(clf, X, y, *, synthetic=True)` — per-species recall / macro-F1 / confusion + CV-CI; гейт честности проставляет `provenance="SYNTHETIC — not a species metric"`, реальное видовое число только с `synthetic=False` на iNatSounds (кластер). TF — только lazy.
- `faun/embeddings` — **единый владелец** батч-экспорта эмбеддингов: протокол `Embedder` + `PerchEmbedder` (1280, 32k/окно 160000), `Perch2Embedder` (**1536**, 32k/160000) и `YamnetEmbedder` (2048, 16k/concat(mean,max)) поверх `experiments/wrappers/{perch,perch_v2,yamnet_probe}` (ленивый TF), `embed_batch`, `EmbeddingCache` (npz). Дублировать embedding-export запрещено.
- `faun/datasets` — `iNatSoundsDataset(root)` (`root/<species>/<clip>`) → `manifest`/`vocab`/`split(seed)`; первый источник с ИСТИННЫМИ видовыми метками (raw180 их не имеет). MINI-фикстура заморожена в `tests/fixtures/inatsounds_mini/README`.
- `faun/labeling` — `batch_label(archive, models, out_jsonl, emb_out=None, embedder=None)`: мультимодельная псевдо-разметка (Perch+BirdNET) → `detections.jsonl` (`model:*`, `pseudo`) + консенсус Perch∩BirdNET + опц. `embeddings.npz`. `training_candidates` — лицензионный гейт: метки `model:birdnet` (CC BY-NC-SA, ShareAlike) НИКОГДА не идут в обучающий набор.
- `faun/health` — `health() -> dict` (status/service/version/jobs_root_writable), pure-stdlib, не падает; `version` из env `FAUN_VERSION` (дефолт `2.0.0-rc`) — `/healthz` отражает билд-тег (blue-green health-gate сверяет); роут `GET /healthz` в `faun/api`.
- `faun/jobs` — изоляция батчей: namespace на `job_id`, `workdir=jobs_root/<job_id>/`, без общих temp-путей.
- `faun/storage` — протокол `Storage` + только `LocalFSStorage` (S3 — задача на июль, НЕ сейчас).
- `faun/api` — FastAPI: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/results.csv`, `GET /jobs`, `GET /jobs/{id}/detections`, `GET /jobs/{id}/segments/{det}.wav`, `POST /jobs/{id}/detections/{det}/label` (flock read-modify-write), `GET /dashboard`, `GET /review`, `GET /healthz` (→ `faun.health.health`). `run_pipeline` пишет `detections.jsonl` + клипы `segments/<det>.wav` из ОРИГИНАЛА на исходной sr (CSV/sidecar — без изменений); чистит `job_dir/_source` в `finally` (disk-leak). Опц. **HTTP Basic Auth** — middleware на ВСЕХ роутах вкл. `/static`, кроме `/healthz` (env `FAUN_BASIC_USER`/`FAUN_BASIC_PASS`; обе пустые → ОТКРЫТО/default-open; `hmac.compare_digest`; 401+`WWW-Authenticate`). Валидация `lat/lon` (WGS84-границы, NaN/Inf→422) и длины меток.
- `faun/cli` — `faun process <dir|URL|Я.Диск> [--out results.csv]`; `faun retrain --labels <csv> --model yamnet|perch|perch-v2 --out probe.pkl` (бэкбон пробы — любой embed-адаптер); `faun export-clips --job <dir> --out clips.zip`; `faun export-labels --job <dir> --out labels.csv` (детекции→retrain-CSV: только `is_ground_truth`-метки, **абсолютный** `segment_path` — закрывает петлю review→retrain); `faun batch-label --archive <dir|URL> --out <jsonl> [--emb-out npz] [--embedder perch|perch-v2|yamnet] [--models perch,birdnet]`; `faun fetch-dataset --root <iNatSounds>`; `faun eval-species --probe <pkl> --dataset <dir> [--embedder perch|perch-v2|yamnet] [--real]` (без `--real` число помечается SYNTHETIC); `faun finetune --dataset <iNatSounds> --out <ckpt_dir> [--model passt|ast|beats]` (РЕАЛЬНЫЙ fine-tune трансформера, только кластер-GPU). `process`/`batch-label` теперь принимают URL/Я.Диск (резолв в `faun.sources`).
- `faun/static` — vanilla-JS UI, 3 окна (имена заморожены): `index.html` (загрузка/очередь), `dashboard.html` (Leaflet-карта ловушек + список job), `review.html` (детекции + аудиоплеер реального клипа + переразметка лесником); общие `app.js`/`styles.css`. Редизайн 2026-06-18: единый «лесной» визуальный стиль, общий header/footer, demo-facing для заказчика.
- `deploy/` — артефакты деплоя v2: `Dockerfile` (slim, TF-free, Stub — **мгновенный откат**), **`Dockerfile.ml`** (TF-образ с РЕАЛЬНЫМ Perch 2: та же digest-база + `requirements-ml.txt` `tensorflow-cpu==2.20.0`+`kagglehub`, копирует `experiments/wrappers` для `_infer`, дефолт `FAUN_CLASSIFIER=perch-v2`+`PERCH_V2_MODEL_PATH=/models/perch2`; модель в томе `faun-models`, не в образе), `docker-compose.yml`, `README.md`. **Задеплоен** на кластер (текущий тег — `faun-api:2.3.0-ml-20260624`, см. Инфраструктуру); собирается на кластере (локально docker нет). `scripts/` — кластерные launch-скрипты: `train_inatsounds.sh`, `batch_label_raw180.sh` (one-command, для ручного запуска на cluster-alex), `extract_inatsounds_subset.py` (курируемый reserve-subset видов из `val.tar.gz`+`val.json`: чистая TF-free `select_targets` + потоковый scan tar → `root/<Genus_species>/<clip>`), `eval_inatsounds_perchv2.py` (честная **held-out** видовая метрика Perch 2: дизъюнктный split → проба на train → `species_eval(synthetic=False)` на val + apples-to-apples zero-shot baseline на тех же видах + деплой-проба на полном наборе; peak-norm serve-parity).
- `faun/sources` — **P0-фикс**: слой резолва источника ПЕРЕД замороженным `ingest.scan`. `resolve_source(src, workdir)` (локальный путь / http(s)-zip / публичная шара Я.Диска, в т.ч. подпапка `…/d/<key>/A1` через `public_key`+`&path`, НЕ вклеивая её в ключ) → локальная папка; `source_provenance` → `results_meta.json`. Раньше `Path("https://…")` схлопывал `//`. Безопасность (merge-blocking): SSRF (резолв IP + блок private/loopback/link-local/CGNAT `100.64/10` = tailnet кластера; ручной обход редиректов с проверкой каждого хопа ДО запроса), zip-bomb (счётчик при распаковке), zip-slip, size-cap в потоке, удаление zip+`_source/`. Env `FAUN_SOURCE_{TIMEOUT_S,MAX_BYTES,MAX_UNCOMPRESSED_BYTES,MAX_ENTRIES,MAX_REDIRECTS}` (дефолты под ~23 ГБ Я.Диск-папки).
- `faun/settings` — `Settings`(frozen) + `get_settings()` (кэш, `cache_clear()` в тестах): единый типизированный конфиг, **читается через `get_settings()`** в `api`/`sources`/адаптерах(`perch`/`perch_v2`/`yamnet`)/`health`; +`basic_user`/`basic_pass` (Basic Auth, verbatim-чтение через `_env_secret_opt`), +`perch_v2_probe_path` (`PERCH_V2_PROBE_PATH`, для `PerchProbeAdapter`). source-лимиты enforce-ит `faun.sources` (uncompressed-cap — через `_int_env` с дефолтом из settings, чтобы env-override и zip-bomb-тест работали). `faun/obs` — `setup_logging(json)` + `with_job_context(job_id)` (структурные JSON-логи, stdlib).
- `faun/training` — РЕАЛЬНЫЙ PyTorch fine-tune аудиотрансформера на iNatSounds (vs замороженная проба в `retraining`): `iNatTorchDataset`/`make_loaders`, `build_backbone` (PaSST 768 по умолчанию / AST / BEATs / numpy-стаб), `SpeciesHead`, `finetune(...)` (freeze→unfreeze, grad-accum, AMP, class-weight, early-stop, чекпойнт+resume), `save/load_checkpoint`. Контрол-флоу тестируется БЕЗ torch (numpy-стаб + инъекция `_backbone`/`_loaders`) + один реальный fwd/bwd под `requires_torch`. **Ни одного module-level `import torch`.** Тяжёлое — `requirements-train.txt` (lazy, кластер `faun-ml-torch`); запуск — `scripts/finetune_inatsounds.sh`, дока — `docs/finetuning.md`.
- `faun/ml` — REUSE-ядро из хакатона: `onset.py`, `ndsi.py`, `yamnet.py`, `datasphere_client.py`.

Контракт интерфейсов заморожен в `faun/INTERFACES.md` — Phase-2 волны пишут против него, сигнатуры не меняем. v2-прототип расширяет контур ТОЛЬКО аддитивно: новый sidecar `detections.jsonl` и новые роуты, без изменения колонок CSV и замороженных сигнатур. `requirements-pipeline.txt` пополнен `scipy`+`scikit-learn` (нужны localization/retraining; тяжёлые ML-либы по-прежнему lazy).

## Вычислитель

Кластер **cluster-alex** (RTX 2060 SUPER 8GB, CUDA 13). TF/JAX без сборки под
cu130 — **CPU-only** на кластере и в CI; GPU доступен через PyTorch-стек.
Docker-образы на кластере: `faun-ml-cpu`, `faun-ml-torch`; данные —
`/home/oleg/faun-data/`. Веса `faun/ml/yamnet_forest_classifier_v7.keras`
(~6.4 MB) в `.gitignore` — раскладываются на кластер вручную, TF импортируется
лениво внутри функций `faun/ml/yamnet.py`. Бенчмарки моделей — `experiments/`
(`python -m experiments.runner`, эксперименты E0..E5, E10).

## Legacy

Весь хакатонный код заморожен в `legacy/` и на теге **v1-hackathon**:
real-time мониторинг, YAMNet-голова, TDOA-триангуляция, LoRa-mesh, Telegram-боты,
дрон (ArduPilot), RAG-агент, Yandex Workflows. Импорты внутри `legacy/` НЕ чиним.
Демо хакатона продолжает крутиться на кластере. Тесты legacy лежат в
`legacy/tests/` и CI их НЕ гоняет (CI запускает только `tests/`).

Полезное из legacy (как справка): `FOLDER_ID=b1g5lqh1mqg84cabtejb`,
`SEARCH_INDEX_ID=fvttk7bjvnm39qogtoep` — относятся к legacy RAG, в v2 не используются.

## Инфраструктура

- Ingress (с 2026-06-18, обновлено 2026-06-19, актуализировано 2026-06-24): `faun.antopkin.ru/` → nginx
  `anchor` (контейнер `delphi-press-nginx-1`) → tailnet `100.64.0.1:8010` = **v2-pipeline** (контейнер
  `faun-api`, образ **`faun-api:2.3.0-ml-20260624`**, `FAUN_CLASSIFIER=perch-v2` = **РЕАЛЬНОЕ
  распознавание видов Perch 2**, не stub; модель в томе `faun-models:/models/perch2`; **HTTP Basic
  Auth ВКЛ**, user=`faun`, кроме `/healthz`). Откат — предыдущий контейнер `faun-api-prev-2.2.0`
  (припаркован stopped: `docker start faun-api-prev-2.2.0`; образ `faun-api:2.2.0-ml-20260619` цел
  на диске). Демо v1-hackathon переехало на `faun.antopkin.ru/v1/` → `100.64.0.1:8003` (`/ws` оставлен
  в корне для live-аудио).
- Авто-деплой ОТКЛЮЧЁН: старый VPS (`213.165.220.144`, delphi-press) удалён
  2026-05-30; `deploy.yml` — заглушка `workflow_dispatch`. CI/CD на кластер —
  июльская задача (GHA-раннеры не видят tailnet).
- **Forgejo на кластере НЕ трогать** — работаем только с GitHub.
- Ветка для коммитов: `main`. CI: `.github/workflows/ci.yml` (`pytest tests/`,
  `bandit faun`, `pip-audit`; `legacy/tests/` не гоняется).

## Инструкции

- Язык общения: русский.
- Модели (`*.keras`, `*.h5`, `*.npz`) в `.gitignore`.
- Зависимости pipeline: `requirements-pipeline.txt`. Legacy-requirements — в `legacy/`.
- **Пиши простым языком.** Не используй термины ради вида умности. Нужен термин и есть сомнение,
  что собеседник его знает — дай короткую ссылку на источник (см. `docs/ГЛОССАРИЙ.md`).
- **Не выдумывай доменные данные** (виды, координаты, чеклисты). Не подавали на вход — значит у нас
  этого нет, так и говорим: «не измерено / на вход не подавали / даёт орнитолог». Метрику на
  публичном датасете (iNatSounds) НЕ называй метрикой «в заповеднике». Подробности — `docs/АУДИТ-2026-06-25.md`.
