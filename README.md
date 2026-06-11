# Faun

Офлайн batch-pipeline распознавания видов птиц по записям с аудиоловушек
(Yandex Cloud + Президентский фонд природы). Двухэтапный контур:
детектор звуковых событий → классификатор видов → CSV.

## Quickstart

```bash
pip install -r requirements-pipeline.txt

# CLI: обработать папку с записями ловушек
faun process <dir> --out results.csv

# API: поднять сервис
uvicorn faun.api:app --reload
#   POST /jobs {source_path|url, lat, lon} -> {job_id}
#   GET  /jobs/{id}            -> status
#   GET  /jobs/{id}/results.csv

# UI: faun/static/index.html (форма -> запуск -> скачивание CSV)
```

CSV-формат: `track,start_sec,duration_sec,species,probability` (+ sidecar с метаданными ловушки).
Контракт интерфейсов: [`faun/INTERFACES.md`](faun/INTERFACES.md). Детали — в Phase 6.

Хакатонный код (real-time, TDOA, LoRa, Telegram, дрон, RAG) заморожен в
[`legacy/`](legacy/) и на теге `v1-hackathon`.
