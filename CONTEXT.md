# Faun — domain lexicon

Назначение: словарь домена для skill'а improve-codebase-architecture и для AI-навигации
по `faun/`. Каждое понятие привязано к `file:symbol`. Цель — устранить терминологические
ловушки (Segment vs Detection, track vs trap_id, размерности эмбеддеров), на которых
агент чаще всего ошибается.

Проект: офлайн batch-pipeline распознавания **видов птиц** по записям с аудиоловушек.
Двухэтапный контур: детектор событий → классификатор видов. Заказчик: Yandex Cloud +
Президентский фонд природы. Бэкенд сейчас — заглушка (`FAUN_CLASSIFIER=stub`).

---

## Segment vs Detection vs «event»

Три слова про одно и то же звуковое явление, но на разных уровнях — путать нельзя.

- **«event»** — неформальное имя акустического явления, найденного onset-детектором
  (`faun.ml.onset.OnsetDetector`). Не тип, не класс. Каждый event превращается в `Segment`.
- **`Segment`** (`faun.segmentation.Segment`, frozen dataclass) — *временно́е окно* на
  таймлайне **исходной** записи: `start_s`, `duration_s` (+ свойство `end_s`). Только время,
  без привязки к ловушке, файлу или клипу. Производится
  `faun.segmentation.SegmentExtractor.extract(waveform, sr) -> list[Segment]` (frozen-контракт,
  `faun/INTERFACES.md`).
- **`Detection`** (`faun.detections.Detection`) — **центральная абстракция** контура. Это
  `Segment`, привязанный к контексту:
  - `trap_id` — ловушка-источник;
  - `source_file` — имя исходной записи;
  - `segment` — сам `Segment` (окно на оригинале);
  - `segment_path` — путь к вырезанному клипу на диске (`segments/<detection_id>.wav`);
  - `labels: list[Label]` — **упорядоченный** список меток (псевдо-метки моделей и
    человеческие правки сосуществуют в одной записи).
  Создаётся через `Detection.new(...)` (свежий `uuid4`-hex id + производный `segment_path`).
  Персист — JSONL, одна детекция на строку, атомарно (`write_detections` / `read_detections`).

Кратко: **event** → детектор находит → **Segment** (только время) → обвешивается контекстом
и метками → **Detection** (то, что хранится и переразмечается).

## trap_id vs «track»

- **`trap_id`** — **авторитетный** идентификатор ловушки. Одна папка = одна ловушка
  (`A1..A5`), значение приходит из `info.txt`/`faun.ingest.scan` → `AudioFileEntry.trap_id`.
  Это и есть «истинное» понятие источника. Хранится в `Detection.trap_id`.
- **«track»** — это **только имя колонки CSV** (`faun.output.COLUMNS`, первый элемент
  `"track"`). Отдельного концепта «track» в домене нет. В текущем выводе колонка `track`
  несёт **значение `trap_id`**: `run_pipeline` пишет строку `{"track": entry.trap_id, ...}`
  (`faun/api.py`, `writer.write_row`). То есть `track` — это сериализованный `trap_id`,
  а не самостоятельная сущность.

Не вводи «track» как доменное понятие при рефакторинге: это легаси-имя выходной колонки.

## Label

`faun.detections.Label` — одна аннотация детекции, кортеж из пяти полей:
`(species, probability, source, status, ts)`.

- **`source`** — провенанс метки. Канонические константы (`faun/detections.py`):
  - `SOURCE_PERCH = "model:perch"`
  - `SOURCE_PERCH_V2 = "model:perch-v2"`
  - `SOURCE_BIRDNET = "model:birdnet"`
  - `SOURCE_YAMNET_PROBE = "model:yamnet-probe"`
  - `SOURCE_STUB = "model:stub"`
  - `SOURCE_EXPERT = "expert:ornithologist"`
  - `SOURCE_RANGER = "operator:ranger"`
- **`status`** — жизненный цикл метки (`faun/detections.py`):
  - `STATUS_PSEUDO = "pseudo"` (метки моделей)
  - `STATUS_CONFIRMED = "confirmed"`
  - `STATUS_CORRECTED = "corrected"` (человеческие правки)
- `probability` — `float | None` (человеческая метка через `Label.now(...)` пишется без
  вероятности).
- Конструкторы: `Label.now(...)` (штамп текущим UTC) и
  `Label.from_prediction(pred, source, status=STATUS_PSEUDO)` (лифт `Prediction` в метку).

## Ground truth

`faun.detections.is_ground_truth(label)` истинна **тогда и только тогда**, когда:
`label.source` начинается на `expert:` или `operator:` (человек) **И**
`label.status ∈ {confirmed, corrected}`.

Псевдо-метки моделей (`model:*`, всегда `pseudo`) **никогда** не ground truth. Единый дом —
`faun.detections` (`is_ground_truth` + `TRAINING_EXCLUDED_SOURCES`); `faun.retraining` и
`faun.labeling` их **импортируют**, не дублируют. `is_ground_truth` принимает и `Label`-объект,
и dict. Это hard-гейт обучения: `faun.retraining.retrain_from_labels` отказывается (`ValueError`)
до любого обращения к модели, если ground-truth-меток нет.

## Контракт входа классификатора (ловушка)

`SpeciesClassifier.classify(segment, sr) -> list[Prediction]` (frozen, `faun/INTERFACES.md`).
Несмотря на имя параметра `segment`, на вход идёт **mono float32 numpy-массив на 16 кГц**,
а **НЕ** объект `Segment`. Реальные адаптеры делают `np.asarray(segment)` — передать туда
`faun.segmentation.Segment` нельзя.

И `run_pipeline` (`faun/api.py`), и `batch_label` (`faun/labeling/__init__.py`) идут через
общий исполнитель `faun.pipeline.run_batch`, который режет клип из оригинала (`slice_clip`),
делает downmix в mono и resample до `CLASSIFY_SR = 16000` (`to_classifier_input`, препроцессинг —
в `faun.audio`). См. `faun.pipeline.to_classifier_input` с явным комментарием на эту ловушку.

## Размерности эмбеддеров (известная ловушка)

`faun.embeddings.Embedder.embed(waveform, sr) -> np.ndarray[DIM]`. DIM **различаются** —
смешение размерностей роняет sklearn невнятной ошибкой:

| Класс | `DIM` | Препроцессинг |
|---|---|---|
| `faun.embeddings.PerchEmbedder` | **1280** | downmix → 32 кГц → окно 160000 |
| `faun.embeddings.Perch2Embedder` | **1536** | downmix → 32 кГц → окно 160000 |
| `faun.embeddings.YamnetEmbedder` | **2048** | downmix → 16 кГц → `concat(mean, max)` |

Отдельно: `faun.classification.YAMNetAdapter.embed(...)` возвращает **1024** (только
`mean`-pooling) — это НЕ то же самое, что `YamnetEmbedder` (2048, `concat(mean,max)`).
`faun.classification.Perch2Adapter.DIM = 1536` (`PERCH_V2_DIM`), он сознательно НЕ
откатывается на Perch 1 (1280), чтобы не подменить размерность молча.

**Инвариант:** обучение и оценка обязаны использовать **один и тот же** эмбеддер.
`faun.retraining.species_eval` enforce-ит dim-гейт: если `clf.n_features_in_ != X.shape[1]`,
бросает `ValueError` с явным текстом про 2048-vs-1024.

## Резолв источника (перед ingest)

`faun.sources.resolve_source(src, workdir)` работает **ПЕРЕД** замороженным
`faun.ingest.scan`: локальный путь / http(s)-zip / публичная шара Я.Диска → локальная папка.
До этого слоя `Path("https://…")` схлопывал `//` (P0-баг). Таксономия отказа —
`faun.sources.SourceError.kind ∈ {bad-scheme, ssrf, not-found, network, too-large, zip-slip,
not-an-archive, empty}`; `faun.api._execute_job` кладёт её в `job.params["error_kind"]`.

## Конфиг

`faun.settings.get_settings()` (кэш `lru_cache`, `cache_clear()` в тестах) — единый
типизированный дом для всех `FAUN_*` ручек (`jobs_root`, `classifier`, `log_json`,
source-лимиты, model-paths). Не читай `os.environ` напрямую в новом коде — иди через
`get_settings()`.

## Правило честности метрик

Синтетические видовые числа помечаются `provenance = "SYNTHETIC — not a species metric"`
(`faun.retraining.species_eval` при `synthetic=True`, дефолт). Реальная видовая метрика
существует **только** при `synthetic=False` на настоящем iNatSounds (на кластере). Не
выдавай синтетическое число за species-метрику.
