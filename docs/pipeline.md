# Faun pipeline (v2)

Офлайн batch-pipeline распознавания видов птиц по записям с пассивных
аудиоловушек. Контракт интерфейсов заморожен в
[`faun/INTERFACES.md`](../faun/INTERFACES.md) — все стадии пишут против него,
сигнатуры не меняются.

## Архитектура

```
                   ┌──────────────────── jobs (uuid-изоляция) ────────────────────┐
                   │  workdir = jobs_root/<job_id>/  •  atomic manifest.json       │
                   │                                                               │
  source_path/url  │   ingest ─► ordering ─► segmentation ─► classification ─► output
  (каталог ловушек)│     │          │            │                │             │   │
                   │   scan      сортировка   downmix→soxr     Protocol      CsvWriter
                   │   info.txt  + gap-детект 48k→16k →onset   Species-      results.csv
                   │   A1..A5    цикла 10м+1м                  Classifier    + results_meta.json
                   │                                                               │
                   └──────────────────── storage (LocalFSStorage) ────────────────┘
                                          S3/Object Storage — июль
```

- **ingest** (`faun/ingest`) — `scan(path) -> Manifest`. Каждая папка = одна
  ловушка (`A1..A5`) со своим `info.txt`. Парсит CSV `info.txt` и timestamp из
  имени файла. Элемент манифеста: `AudioFileEntry(path, trap_id, start_dt,
  lat, lon, meta, duration_s, sr)`.
- **ordering** (`faun/ordering`) — хронологическая сортировка записей манифеста
  и детект пропусков цикла записи (10 мин записи + 1 мин паузы).
- **segmentation** (`faun/segmentation`) — `SegmentExtractor.extract(waveform,
  sr) -> list[Segment]`. Внутри: downmix в моно, ресемпл `soxr` 48k→16k,
  onset-детектор (`faun/ml/onset.py`). `Segment(start_s, duration_s)`.
- **classification** (`faun/classification`) — протокол `SpeciesClassifier`
  (`classify(segment, sr) -> list[Prediction]`), `Prediction(species,
  probability)`. Адаптеры — ниже.
- **output** (`faun/output`) — `CsvWriter` пишет `results.csv` и sidecar
  `results_meta.json`.
- **jobs** (`faun/jobs`) — изоляция батчей: namespace на `job_id`,
  `workdir=jobs_root/<job_id>/`, atomic `manifest.json`, без общих temp-путей.
- **storage** (`faun/storage`) — протокол `Storage` (put/get/url). Сейчас
  только `LocalFSStorage`; S3 / Object Storage — июльская задача.

Точки входа: `faun/api.py` (FastAPI + web-UI на `/`) и `faun/cli.py`
(`python -m faun.cli process <dir> [--out results.csv]`). Обе синхронно
вызывают `faun.api.run_pipeline(job_dir, source_path, lat, lon)`.

## CSV-схема

Порядок колонок заморожен (`faun/INTERFACES.md`, `faun/output.COLUMNS`):

| Колонка        | Тип   | Описание                                   |
|----------------|-------|--------------------------------------------|
| `track`        | str   | Идентификатор источника (имя WAV / трек)   |
| `start_sec`    | float | Начало сегмента, сек (округление 2 знака)  |
| `duration_sec` | float | Длительность сегмента, сек (2 знака)        |
| `species`      | str   | Вид (или `unknown`)                        |
| `probability`  | float | Вероятность (округление 4 знака)           |

Пример:

```
track,start_sec,duration_sec,species,probability
A1_20260612_063000.wav,12.40,2.10,Turdus merula,0.9100
A1_20260612_063000.wav,31.05,1.80,unknown,0.4200
```

Sidecar `results_meta.json` (имя = `<stem>_meta.json` рядом с CSV) несёт
provenance ловушки:

```json
{
  "trap_id": "A1",
  "lat": 57.3697,
  "lon": 44.6200,
  "files": ["A1_20260612_063000.wav"],
  "pipeline_version": "2.0"
}
```

## Формат info.txt

Одна папка ловушки = один `info.txt` (CSV). Колонки:

```
date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel
```

Пример строки:

```
2026-06-12,06:30:00,44.6200,57.3697,87,14.2,71,A1_20260612_063000.wav,48000,12,L
```

Timestamp записи дополнительно парсится из имени файла; координаты ловушки
(`long`,`lat`) попадают в `AudioFileEntry` и в sidecar.

## Адаптеры классификации и их лицензии

Адаптеры реализуют протокол `SpeciesClassifier`. Тяжёлые ML-зависимости
импортируются лениво — `import faun.classification` их не тянет (PEP 562
`__getattr__`).

| Адаптер          | Модуль                          | Лицензия модели                       | Заметка |
|------------------|---------------------------------|----------------------------------------|---------|
| `StubAdapter`    | `faun/classification/__init__`  | —                                      | Детерминированная заглушка без ML; держит pipeline и тесты независимыми от тяжёлых либ. |
| `BirdNETAdapter` | `faun/classification/birdnet`   | **CC BY-NC-SA** (non-commercial, ShareAlike) | Не пригоден для коммерческого продукта из-за NC. |
| `PerchAdapter`   | `faun/classification/perch`     | **Apache 2.0** (Perch 2)               | Рекомендован как продуктовый. Perch 1 — на TFHub без авторизации. |
| `YAMNetAdapter`  | `faun/classification/yamnet`    | Apache 2.0 (YAMNet)                     | Embeddings + probe, НЕ хакатонная YAMNet-голова из `legacy/`. |

## Эксперименты на кластере

Бенчмарки моделей живут в `experiments/` и гоняются на кластере **cluster-alex**
(RTX 2060 SUPER 8 GB, CUDA 13). Раннер изолирует каждый эксперимент в subprocess
с жёстким timeout; ошибка/таймаут одного не валит очередь, в `results.csv`
пишется строка со статусом `error`/`timeout`/`skip`.

```bash
# отдельные эксперименты
python -m experiments.runner E1 E3

# все exp_e*.py по порядку
python -m experiments.runner --all

# с параметрами
python -m experiments.runner E1 --timeout-min 10 --data-root /home/oleg/faun-data
```

Доступные эксперименты: `E0..E5`, `E10` (модули `experiments/exp_e<N>.py`,
каждый с `run(cfg) -> dict`). Данные по умолчанию — `/home/oleg/faun-data`
(`FAUN_DATA_ROOT`); пик VRAM снимается через `nvidia-smi` (если доступен).
Wrappers моделей — в `experiments/wrappers/` (birdnet, perch, clap,
yamnet_probe).

> TensorFlow/JAX без сборки под CUDA 13 (cu130) на кластере работают CPU-only;
> GPU доступен через PyTorch-стек.
