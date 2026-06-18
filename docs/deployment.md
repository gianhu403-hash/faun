# Деплой и инфраструктура

## Текущее состояние

- **Faun v2 (batch-pipeline)** — пока без авто-деплоя. Запускается локально или
  на кластере вручную (см. ниже). Артефакты деплоя СТЕЙДЖ-готовы в `deploy/`
  (Dockerfile + docker-compose), но образ собирается руками — локально docker не
  гоняется.
- **Замороженное демо v1-hackathon** продолжает крутиться на кластере и
  доступно через <https://faun.antopkin.ru>. Ingress:
  `anchor` nginx → tailnet `100.64.0.1:8003`. Это демо хакатонного контура
  (real-time мониторинг, TDOA, LoRa, Telegram, дрон, RAG), не v2-pipeline.
- **Авто-деплой ОТКЛЮЧЁН.** Старый VPS (`213.165.220.144`, shared
  `delphi-press`) удалён **2026-05-30** — больше не существует.
  `.github/workflows/deploy.yml` — заглушка `workflow_dispatch` (ручной триггер,
  ничего не деплоит): GHA-раннеры не видят tailnet кластера, поэтому
  автоматический pull на кластер невозможен до настройки self-hosted runner /
  ssh-jump (июль).
- **CI** работает: `.github/workflows/ci.yml` гоняет `pytest tests/`,
  `bandit faun`, `pip-audit`. CI гоняет только каталог `tests/`; `legacy/tests/`
  игнорируется.

## Health-эндпоинт

- `GET /healthz` — лёгкий liveness/readiness без тяжёлых импортов. Handler —
  `faun/health.py` (`health() -> dict`), чистый stdlib, TF-free; роут подключён
  в `faun/api.py`.
- Payload: `{"status": "ok"|"degraded", "service": "faun-api", "version":
  "2.0.0-rc", "jobs_root_writable": bool}`. Если `FAUN_JOBS_ROOT` недоступен на
  запись — `status` деградирует до `degraded`, handler при этом НЕ падает.
- `HEALTHCHECK` в `deploy/Dockerfile` и `deploy/docker-compose.yml` дёргает
  `/healthz` каждые 30s.

## Артефакты деплоя (`deploy/`)

| Файл                        | Назначение                                          |
|-----------------------------|-----------------------------------------------------|
| `deploy/Dockerfile`         | Лёгкий образ `python:3.12-slim`, только `requirements-pipeline.txt`, **без TF**; запуск `uvicorn faun.api:app` на порту 8010, непривилегированный пользователь, `HEALTHCHECK` на `/healthz`. |
| `deploy/docker-compose.yml` | Один сервис `faun-api`, порт `8010:8010`, volume `faun-jobs:/data/jobs`, `FAUN_JOBS_ROOT`. |
| `deploy/README.md`          | Runbook: build/run команды, env, выбор порта, ручной деплой. |

TF/JAX в образе намеренно отсутствуют: API TF-free на Stub-классификаторе.
Реальные модели (Perch/BirdNET/YAMNet-probe) поднимаются на кластере в образах
`faun-ml-cpu`/`faun-ml-torch`, не в этом образе.

> **Сборка СТЕЙДЖ.** Локально docker не запускается, поэтому образ только описан,
> но не собран. Сборка/запуск — вручную на целевом хосте.

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
uvicorn faun.api:app --reload        # UI на http://127.0.0.1:8000/
curl -fsS http://127.0.0.1:8000/healthz
```

`requirements-pipeline.txt` намеренно лёгкий (fastapi, uvicorn, numpy,
soundfile, soxr, pydantic, scipy, scikit-learn, …) — **без tensorflow**: тяжёлые
модели тянутся лениво только внутри адаптеров классификации.

### Через образ деплоя

```bash
# из корня репозитория (на целевом хосте, не локально)
docker compose -f deploy/docker-compose.yml up -d --build
curl -fsS http://localhost:8010/healthz
```

### На кластере (ML-образы с данными)

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

## Порт v2

v2-API слушает **8010**.

> ⚠️ Порты **8003 / 8005 / 9000** на кластере заняты замороженным демо
> v1-hackathon — НЕ переиспользовать. 8010 выбран как свободный; целевой
> host/port пилота **не подтверждён** (open question), 8010 — предложение по
> умолчанию.

## Деплой на кластер — РУЧНОЙ

> **Авто-деплоя НЕТ.** Фактический деплой v2 на кластер — ручная утренняя/
> июльская задача. `deploy.yml` — заглушка `workflow_dispatch`.

nginx на хосте `anchor` — read-only bind-mount (правки через write +
force-recreate, не sed/reload). Forgejo на кластере не трогаем — только GitHub.

## July roadmap

- **CI/CD на кластер** — self-hosted GHA runner внутри tailnet (или ssh-jump),
  чтобы push в `main` доезжал до cluster-alex. Сейчас GHA-раннеры tailnet не
  видят.
- **S3 / Object Storage** — реализация `S3Storage` против протокола
  `faun/storage.Storage` (сейчас только `LocalFSStorage`).
- **Веб-эндпоинт пилота** — публичный доступ к v2-API для оператора (поверх
  существующего FastAPI + UI на `/`), с подтверждённым host/port и ingress.

## История (удалённая инфраструктура)

Старый продакшен v1 жил на shared VPS `delphi-press` (`213.165.220.144`,
Debian 12). **Хост удалён 2026-05-30.** Авто-деплой туда (push → GHA → ssh →
`docker compose -p faun up -d --build`), CSP-конфиги
(`faun-security-headers.conf`), nginx-проброс — всё относилось к этому VPS и
больше не действует. Здесь оставлено только как контекст истории, не как рабочая
инструкция.
