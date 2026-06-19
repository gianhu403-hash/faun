# ADR 0003 — Reusable pipeline executor + vocab/dim contract (`faun/pipeline.py`)

- Status: Proposed
- Date: 2026-06-19
- Deciders: Faun v2 pipeline team
- Supersedes: —

## Context

Ядро контура «для каждой записи манифеста: прочитать сигнал → выделить сегменты →
вырезать клип на исходной sr → downmix+resample до 16 кГц → классифицировать → собрать
`Detection`» **продублировано** в двух местах:

1. `faun.api.run_pipeline` (`faun/api.py`) — один классификатор; пишет CSV
   (`results.csv` + `results_meta.json`), клипы `segments/<detection_id>.wav` из
   **оригинала на исходной sr** и `detections.jsonl`.
2. `faun.labeling.batch_label` (`faun/labeling/__init__.py`) — мультимодельная разметка
   (Perch+BirdNET); опционально экспортирует эмбеддинги (`embeddings.npz`), где клипы
   **обязаны оставаться выровненными по строкам с детекциями**.

Обе функции независимо реализуют один и тот же цикл: `sf.read` → `SegmentExtractor.extract`
→ срез клипа на исходной sr → mono+resample до `CLASSIFY_SR = 16000` → `classify` →
`Detection.new(...)`. Логика среза и приведения ко входу классификатора уже разъезжается:
в `run_pipeline` это inline `soxr.resample(...)`, в `batch_label` — выделенные
`_slice_segment` / `_to_classifier_input`.

Критичный инвариант, на котором держится экспорт эмбеддингов в `batch_label`: список
`clips` собирается в **том же** проходе, что и `detections`, поэтому `clips[i]`
соответствует `detections[i]`, и `EmbeddingCache(embeddings=..., ids=[det.detection_id ...])`
выровнен по строкам. Любой рефакторинг, который рассинхронит эти два списка, молча
испортит обучающий набор (эмбеддинг одной детекции уедет на id другой).

## Decision

Ввести `faun/pipeline.py` — переиспользуемый **executor** общего ядра. Он принимает
манифест (или путь, уже резолвнутый `faun.sources.resolve_source`) и классификатор(ы),
прогоняет цикл `read → extract → slice → 16k → classify → build Detection` и возвращает
детекции **вместе с по-детекционными клипами, выровненными по строкам**:
`detections[i]` ↔ `clips[i]`.

Правила:

1. **`run_pipeline` и `batch_label` сохраняют свои замороженные сигнатуры** и становятся
   тонкими обёртками над executor'ом:
   - `run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None) -> Path`
   - `batch_label(archive, models, out_jsonl, emb_out=None, embedder=None) -> dict`

   Никакого signature drift: внешний контракт (`faun/INTERFACES.md`, API-роуты, CLI)
   не меняется. `run_pipeline` поверх детекций по-прежнему пишет CSV + клипы + JSONL;
   `batch_label` — JSONL + опц. эмбеддинги.

2. **Executor гарантирует выравнивание:** `clips[i]` соответствует `detections[i]`. Это
   и есть контракт выравнивания, на который опирается экспорт эмбеддингов в `batch_label`
   (`ids = [det.detection_id ...]`, строки `embeddings` выровнены с `detections`).
   Инвариант становится **тестируемым свойством** executor'а, а не неявным следствием
   «собираем в одном цикле» в двух разных функциях.

3. **Контракт vocab/dim (документируется здесь, enforce — в коде):**
   - `faun.embeddings.Embedder.DIM` — заявленная размерность эмбеддера
     (`PerchEmbedder=1280`, `Perch2Embedder=1536`, `YamnetEmbedder=2048`).
   - `faun.retraining.species_eval` бросает `ValueError`, когда `clf.n_features_in_ != X.shape[1]`
     — то есть обучение и оценка обязаны идти через **один и тот же** эмбеддер. Известная
     ловушка: `YamnetEmbedder=2048` (concat(mean,max)) vs `YAMNetAdapter.embed=1024` (mean).
   - При экспорте эмбеддингов `batch_label` использует тот эмбеддер, что передан оператором
     (`--embedder perch|perch-v2|yamnet`); пустой батч получает форму `(0, embedder.DIM)`
     через `faun.embeddings._embedder_dim`.

## Consequences

Положительно:

- Единственный путь `segment → classify`: цикл `read/extract/slice/16k/classify/build`
  живёт в одном месте, чинится один раз.
- Выравнивание `clips[i]` ↔ `detections[i]` — явный тестируемый инвариант executor'а, а не
  хрупкое совпадение порядка в двух функциях; экспорт эмбеддингов перестаёт зависеть от
  дисциплины конкретного цикла.
- Нет signature drift: `run_pipeline` и `batch_label` сохраняют замороженные сигнатуры,
  внешние контракты не трогаются.
- Контракт vocab/dim задокументирован рядом с executor'ом, который его потребляет; dim-гейт
  `species_eval` получает явную ссылку.

Отрицательно / издержки:

- Ещё один модуль на пути исполнения; `run_pipeline`/`batch_label` становятся обёртками
  (небольшой indirection ради дедупликации).
- Executor должен обслужить два режима — один классификатор (`run_pipeline`) и несколько
  (`batch_label`). Если параметризовать неаккуратно, легко получить «God-функцию»; держать
  поверхность узкой (вход: манифест + модели; выход: детекции + выровненные клипы), запись
  артефактов (CSV/JSONL/npz) оставить вызывающим.

## References

- `faun/api.py` — `run_pipeline` (single classifier; CSV + клипы + `detections.jsonl`).
- `faun/labeling/__init__.py` — `batch_label` (multi-model; `_slice_segment`,
  `_to_classifier_input`, выровненный `clips`-список → `EmbeddingCache`).
- `faun/segmentation/__init__.py` — `SegmentExtractor.extract` (источник `Segment`).
- `faun/retraining.py` — `species_eval` (dim-гейт `n_features_in_` vs `X.shape[1]`).
- `faun/embeddings.py` — `Embedder.DIM`, `_embedder_dim`, `embed_batch`.
- `faun/INTERFACES.md` — замороженные сигнатуры `run_pipeline` (косвенно через API) и
  `batch_label`.
