# Деплой и инфраструктура

## Текущее состояние

- **Faun v2 (pipeline)** — пока без авто-деплоя. Запускается локально или на
  кластере вручную (см. ниже). Веб-эндпоинт пилота — июльская задача.
- **Замороженное демо v1-hackathon** продолжает крутиться на кластере и
  доступно через <https://faun.antopkin.ru>. Ingress:
  `anchor` nginx → tailnet `100.64.0.1:8003`. Это демо хакатонного контура
  (real-time мониторинг, TDOA, LoRa, Telegram, дрон, RAG), не v2-pipeline.
- **Авто-деплой ОТКЛЮЧЁН.** Старый VPS (`213.165.220.144`, shared
  `delphi-press`) удалён **2026-05-30** — больше не существует, упоминания о нём
  в истории помечены как удалённый хост. `.github/workflows/deploy.yml` —
  заглушка `workflow_dispatch` (ручной триггер, ничего не деплоит): GHA-раннеры
  не видят tailnet кластера, поэтому автоматический pull на кластер невозможен
  до настройки self-hosted runner / ssh-jump (июль).
- **CI** работает: `.github/workflows/ci.yml` гоняет `pytest tests/`,
  `bandit faun`, `pip-audit`. CI гоняет только каталог `tests/`; `legacy/tests/`
  игнорируется.

## Вычислитель: cluster-alex

| Параметр       | Значение                                             |
|----------------|------------------------------------------------------|
| GPU            | RTX 2060 SUPER, 8 GB                                  |
| CUDA           | 13                                                    |
| TF / JAX       | без сборки под cu130 → **CPU-only**                   |
| PyTorch-стек   | использует GPU                                        |
| Каталог данных | `/home/oleg/faun-data/` (`FAUN_DATA_ROOT`)           |
| Docker-образы  | `faun-ml-cpu`, `faun-ml-torch`                        |

`faun-ml-cpu` — лёгкий контур (TF/JAX CPU-only), `faun-ml-torch` — PyTorch с GPU
для тяжёлых моделей (Perch, эксперименты).

## Запуск pipeline

### Локально

```bash
pip install -r requirements-pipeline.txt

# CLI
python -m faun.cli process <dir> --out results.csv

# API + UI
uvicorn faun.api:app --reload   # UI на http://127.0.0.1:8000/
```

`requirements-pipeline.txt` намеренно лёгкий (fastapi, uvicorn, numpy,
soundfile, soxr, pydantic, aiofiles, httpx) — **без tensorflow**: тяжёлые модели
тянутся лениво только внутри адаптеров классификации.

### На кластере

Запуск в одном из ML-образов с примонтированными данными:

```bash
# pipeline (CPU-стек достаточно для ingest/segmentation/stub)
docker run --rm -v /home/oleg/faun-data:/data faun-ml-cpu \
    python -m faun.cli process /data/<trap_dir> --out /data/results.csv

# тяжёлые адаптеры / эксперименты (GPU)
docker run --rm --gpus all -v /home/oleg/faun-data:/data faun-ml-torch \
    python -m experiments.runner --all --data-root /data
```

Эксперименты и работа с данными на кластере подробнее — в
[`pipeline.md`](pipeline.md#эксперименты-на-кластере).

## July roadmap

- **CI/CD на кластер** — self-hosted GHA runner внутри tailnet (или ssh-jump),
  чтобы push в `main` доезжал до cluster-alex. Сейчас GHA-раннеры tailnet не
  видят.
- **S3 / Object Storage** — реализация `S3Storage` против протокола
  `faun/storage.Storage` (сейчас только `LocalFSStorage`).
- **Веб-эндпоинт пилота** — публичный доступ к v2-API для оператора (поверх
  существующего FastAPI + UI на `/`).

## История (удалённая инфраструктура)

Старый продакшен v1 жил на shared VPS `delphi-press` (`213.165.220.144`,
Debian 12). **Хост удалён 2026-05-30.** Авто-деплой туда (push → GHA → ssh →
`docker compose -p faun up -d --build`), CSP-конфиги
(`faun-security-headers.conf`), nginx-проброс — всё относилось к этому VPS и
больше не действует. Здесь оставлено только как контекст истории, не как рабочая
инструкция.
