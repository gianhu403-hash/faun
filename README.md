# Faun

Офлайн batch-pipeline распознавания видов птиц по записям с пассивных
аудиоловушек. Пилот: Yandex Cloud + Президентский фонд природы (активная фаза
с июля). Контур: ingest каталога ловушек → упорядочивание записей →
сегментация по onset → классификация видов → CSV.

## Что делает

Оператор отдаёт системе каталог записей с ловушек (локальный путь или URL).
Pipeline сканирует папки ловушек (`A1..A5`, каждая со своим `info.txt`),
упорядочивает записи хронологически с детектом пропусков цикла, режет каждую
запись на сегменты по onset-детектору, классифицирует сегменты адаптером
видового классификатора и пишет плоский CSV с треками, таймкодами, видом и
вероятностью. Рядом кладётся sidecar `results_meta.json` с метаданными ловушки.

## Quickstart

```bash
pip install -r requirements-pipeline.txt

# CLI: обработать каталог ловушек
python -m faun.cli process <dir> --out results.csv

# API + web-UI
uvicorn faun.api:app --reload
#   GET  /                      one-page UI (faun/static/index.html)
#   POST /jobs {source_path|url, lat?, lon?}  -> {job_id}
#   GET  /jobs/{id}             -> status
#   GET  /jobs/{id}/results.csv -> CSV
```

UI на `/` — форма (путь/URL) → запуск задачи → опрос статуса → таблица и
скачивание CSV.

### Формат CSV

```
track,start_sec,duration_sec,species,probability
A1_20260612_063000.wav,12.40,2.10,Turdus merula,0.9100
```

Колонки фиксированы контрактом (`faun/INTERFACES.md`); рядом —
`results_meta.json` (trap_id, координаты, список файлов, версия pipeline).

## Структура репозитория

```
faun/         пакет pipeline: ingest, ordering, segmentation,
              classification, output, jobs, storage, api.py, cli.py, ml/
experiments/  раннер и скрипты E0..E5, E10 — бенчмарки моделей на кластере
legacy/       замороженный хакатонный код v1 (real-time, TDOA, LoRa,
              Telegram, дрон, RAG); CI его не гоняет
docs/         документация (pipeline.md, deployment.md, legal/, ...)
tests/        pytest-сьют pipeline (CI запускает только этот каталог)
```

## Документация

- Архитектура и форматы: [`docs/pipeline.md`](docs/pipeline.md)
- Замороженный контракт интерфейсов: [`faun/INTERFACES.md`](faun/INTERFACES.md)
- Деплой и состояние инфраструктуры: [`docs/deployment.md`](docs/deployment.md)

## Legacy v1-hackathon

Хакатонный код (real-time мониторинг нарушений: YAMNet-голова, TDOA, LoRa,
Telegram-боты, дрон, RAG-агент) заморожен в [`legacy/`](legacy/) и на теге
`v1-hackathon`. Замороженное демо v1 продолжает крутиться на кластере и доступно
через <https://faun.antopkin.ru>.
