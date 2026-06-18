# Faun v2 API — деплой (runbook)

Лёгкий образ batch-pipeline распознавания видов птиц. **Без TensorFlow**: образ
ставит только `requirements-pipeline.txt` и работает на Stub-классификаторе
(API TF-free). Реальные модели (Perch/BirdNET/YAMNet-probe) поднимаются на
кластере `cluster-alex` в образах `faun-ml-cpu`/`faun-ml-torch`, не здесь.

## Сборка и запуск

> **Образ собирается ВРУЧНУЮ на целевом хосте.** Локально docker не гоняется,
> CI/CD на кластер не настроен (см. ниже).

```bash
# из корня репозитория
docker compose -f deploy/docker-compose.yml up -d --build

# или напрямую
docker build -f deploy/Dockerfile -t faun-api:2.0.0-rc .
docker run -d --name faun-api -p 8010:8010 \
    -e FAUN_JOBS_ROOT=/data/jobs \
    -v faun-jobs:/data/jobs \
    faun-api:2.0.0-rc

# проверка живости
curl -fsS http://localhost:8010/healthz
# -> {"status":"ok","service":"faun-api","version":"2.0.0-rc","jobs_root_writable":true}
```

## Переменные окружения

| Переменная        | Значение по умолчанию | Назначение                                   |
|-------------------|-----------------------|----------------------------------------------|
| `FAUN_JOBS_ROOT`  | `/data/jobs`          | Каталог job-ов (примонтирован как volume).    |
| `FAUN_CLASSIFIER` | `stub`                | Классификатор: `stub`/`perch`/`birdnet`/`yamnet` (тяжёлые — только на кластере). |

## Порт

v2-API слушает **8010**.

> ⚠️ Порты **8003 / 8005 / 9000** на кластере заняты **замороженным демо
> v1-hackathon** — НЕ переиспользовать. 8010 выбран как свободный.
>
> Целевой host/port пилота **не подтверждён** (open question) — 8010 это
> предложение по умолчанию.

## Health

`GET /healthz` — лёгкий liveness/readiness без тяжёлых импортов (handler в
`faun/health.py`). При недоступном на запись `FAUN_JOBS_ROOT` статус
деградирует до `degraded` (handler не падает). `HEALTHCHECK` в Dockerfile и
compose дёргает этот эндпоинт каждые 30s.

## Деплой на кластер — РУЧНОЙ

> **Авто-деплоя НЕТ.** Фактический деплой на кластер — ручная утренняя/июльская
> задача. `.github/workflows/deploy.yml` — заглушка `workflow_dispatch`,
> ничего не деплоит.
>
> CI/CD на кластер — **июльская задача**: GitHub Actions-раннеры не видят
> tailnet кластера, нужен self-hosted runner / ssh-jump. До тех пор образ
> собирается и поднимается на хосте руками.

Контекст инфраструктуры: ingress на хосте `anchor` — read-only bind-mount
nginx (правки через write + force-recreate, не sed/reload). Forgejo на кластере
не трогаем — работаем только с GitHub.
