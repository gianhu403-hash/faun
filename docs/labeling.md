# Батч-разметка (`faun.labeling`)

Офлайн-проход по архиву ловушек: каждое звуковое событие получает
**псевдо-метки** от моделей-классификаторов и (опционально) один эмбеддинг.
Результат — материал для эксперта-орнитолога и для дообучения продовой головы.

## Что делает `batch_label`

```python
from faun.labeling import batch_label
from faun.classification import PerchAdapter, BirdNETAdapter
from faun.embeddings import PerchEmbedder

summary = batch_label(
    archive="/data/raw180",
    models={"perch": PerchAdapter(), "birdnet": BirdNETAdapter()},
    out_jsonl="/out/detections.jsonl",
    emb_out="/out/embeddings.npz",   # опционально
    embedder=PerchEmbedder(),        # нужен только при emb_out
)
```

Контур:

1. `faun.ingest.scan(archive)` — скан папок ловушек (A1..A5) + `info.txt`.
2. `SegmentExtractor().extract(...)` — onset-сегменты на исходной частоте.
3. На каждый сегмент каждая модель из `models` даёт `list[Prediction]`.
   Каждое предсказание превращается в `Label` с `source=model:<name>`,
   `status=pseudo` и складывается в `Detection`.
4. `detections.jsonl` пишется всегда (через `faun.detections.write_detections`),
   в том числе пустой, если событий нет — без падения.
5. Если заданы и `emb_out`, и `embedder` — пишется `embeddings.npz`
   (один эмбеддинг на детекцию, выровнен по `detection_id`).

Возвращается сводка:

```json
{
  "counts": {"perch": 312, "birdnet": 298},
  "n_detections": 180,
  "n_consensus": 144,
  "paths": {"detections": "/out/detections.jsonl",
            "embeddings": "/out/embeddings.npz"}
}
```

## Консенсус Perch ∩ BirdNET

Детекция считается **консенсусной**, если один и тот же вид предсказали
**обе** модели — `perch` И `birdnet` — для этого сегмента. Консенсус — это
сигнал приоритета для эксперта (две независимые модели согласны), а не метрика
качества. Если хотя бы одна модель молчит на детекции, консенсуса нет.

## Экспорт эмбеддингов

При `emb_out` + `embedder` каждый клип детекции прогоняется через эмбеддер и
складывается в `EmbeddingCache` (`.npz`): матрица `[N, DIM]` плюс `ids`
(`detection_id` построчно).

| Эмбеддер         | DIM  | Частота | Пуллинг              |
|------------------|------|---------|----------------------|
| `PerchEmbedder`  | 1280 | 32 кГц  | окно 160000 (5 с)    |
| `YamnetEmbedder` | 2048 | 16 кГц  | concat(mean, max)    |

> Любое число, полученное из **синтетических** эмбеддингов (тесты/фикстуры),
> помечается `SYNTHETIC — not a species metric` — это не качество классификации
> видов. Реальные видовые метрики считаются только на кластере на реальном звуке.

## ⚠️ Лицензионный гейт BirdNET (HARD)

**BirdNET распространяется под CC BY-NC-SA:**

- **NC (non-commercial)** — некоммерческое использование;
- **SA (ShareAlike)** — производные работы наследуют ту же лицензию.

**ShareAlike "заражает" дообученную голову:** если псевдо-метки BirdNET попадут
в обучающий набор продовой модели, на эту модель ляжет CC BY-NC-SA — что
несовместимо с коммерческим продуктом (Yandex Cloud + Президентский фонд).

**Поэтому псевдо-метки BirdNET — ТОЛЬКО для инвентаризации и приоритизации
эксперту. НИКОГДА в обучающий набор коммерческой модели.**

Гейт зашит в коде:

```python
from faun.labeling import training_candidates
candidates = training_candidates(detections)  # любые model:birdnet вычищены
```

`training_candidates` возвращает детекции, из которых удалены все метки
`source=model:birdnet`; детекции без единой допустимой метки выпадают. Тест
`tests/test_labeling.py::TestBirdnetLicenseGate` — merge-blocker: проверяет, что
ни одна метка BirdNET не доживает до кандидатов на обучение.

Продуктовая модель дообучается на **Perch (Apache-2.0)** псевдо-метках и/или
человеческих метках (`expert:` / `operator:` со статусом `confirmed`/`corrected`,
см. `faun.retraining`). BirdNET остаётся вторым мнением для оператора.

## Как запустить

### Локально (TF-free, фикстура)

TensorFlow локально нет — реальные Perch/BirdNET не запускаются. Контур
проверяется на крошечной фикстуре `tests/fixtures/traps_mini` (две ловушки) с
управляемыми моделями-заглушками и фейковым эмбеддером:

```bash
python -m pytest tests/test_embeddings.py tests/test_labeling.py -q
```

Препроцессинг адаптеров (resample + pad/truncate + downmix) тестируется
TF-free через монкипатч `experiments.wrappers.perch.embed` /
`experiments.wrappers.yamnet_probe.embed_waveform` на stub, который ASSERT-ит
полученную форму и частоту — то есть прогоняется реальный код препроцессинга
без TensorFlow.

### На кластере (cluster-alex, утренний прогон)

```bash
bash scripts/batch_label_raw180.sh
```

Скрипт запускает Perch+BirdNET в образе `faun-ml-cpu` по
`/home/oleg/faun-data/raw180`, пишет `detections.jsonl` + `embeddings.npz`.
Точная `docker run ...` инвокация — в шапке скрипта. TF тянется лениво внутри
wrapper'ов, только на кластере.
