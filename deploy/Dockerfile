# Faun v2 API — лёгкий образ batch-pipeline (без TensorFlow).
#
# TF/JAX намеренно ОТСУТСТВУЮТ: образ ставит только requirements-pipeline.txt и
# работает на Stub-классификаторе (API TF-free). Реальные модели (Perch/BirdNET/
# YAMNet-probe) поднимаются на кластере cluster-alex в образах faun-ml-cpu/torch,
# а не в этом образе.
#
# Сборка СТЕЙДЖ: локально docker не гоняется, образ собирается вручную утром/в
# июле на целевом хосте (см. deploy/README.md).

FROM python:3.12-slim

# Без .pyc и с небуферизованным stdout — чистые логи в контейнере.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAUN_JOBS_ROOT=/data/jobs

WORKDIR /app

# curl нужен для HEALTHCHECK; ставим до зависимостей, чтобы слой кэшировался.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала только requirements — слой переиспользуется при изменении кода.
COPY requirements-pipeline.txt ./
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Затем сам пакет.
COPY faun/ ./faun/

# Непривилегированный пользователь + каталог job-ов (volume в compose).
RUN useradd --create-home --uid 10001 faun \
    && mkdir -p /data/jobs \
    && chown -R faun:faun /app /data
USER faun

EXPOSE 8010

# v2 слушает 8010 — порты 8003/8005/9000 заняты замороженным демо v1 на кластере.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8010/healthz || exit 1

CMD ["uvicorn", "faun.api:app", "--host", "0.0.0.0", "--port", "8010"]
