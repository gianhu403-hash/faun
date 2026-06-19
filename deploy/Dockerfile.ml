# Faun v2 API — ML образ с РЕАЛЬНЫМ Perch 2 (TensorFlow CPU).
#
# В отличие от slim deploy/Dockerfile (TF-free, Stub-классификатор = мгновенный
# откат), этот образ ПЕЧЁТ TensorFlow + Perch 2-инференс прямо в веб-контейнер:
# тот же порт 8010, тот же blue-green деплой. Сама модель Perch 2 (SavedModel,
# ~400 МБ весов) в образ НЕ копируется — она монтируется томом faun-models в
# /models/perch2 (PERCH_V2_MODEL_PATH) и качается на кластер один раз kagglehub.
#
# Сборка — на кластере cluster-alex (CPU-only TF; локально docker не гоняется):
#   docker build -f deploy/Dockerfile.ml -t faun-api:<tag> --build-arg FAUN_VERSION=<tag> .

# Та же digest-пиновка базы, что и в slim deploy/Dockerfile (воспроизводимый
# билд, идентичный рантайм Python). python:3.12-slim linux/amd64, 2026-06-19.
FROM python:3.12-slim@sha256:c2d8472b831337ab296a8ce652e1ba786e9e3034fc445dc58b50a7f5251f0003

# Build-time версия -> в образ, чтобы /healthz отражал тег билда (blue-green
# health-gate сверяет version == задеплоенный тег).
ARG FAUN_VERSION=dev
ENV FAUN_VERSION=$FAUN_VERSION

# Дефолт классификатора этого образа — РЕАЛЬНЫЙ Perch 2 (а не stub); модель
# берётся из тома по PERCH_V2_MODEL_PATH. Эти ENV можно переопределить в
# `docker run -e ...`, но по умолчанию образ распознаёт виды.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAUN_JOBS_ROOT=/data/jobs \
    FAUN_CLASSIFIER=perch-v2 \
    PERCH_V2_MODEL_PATH=/models/perch2

WORKDIR /app

# curl нужен для HEALTHCHECK; ставим до зависимостей, чтобы слой кэшировался.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала только requirements — слой переиспользуется при изменении кода.
# Лёгкий pipeline-стек + TF/kagglehub (тяжёлый слой, кэшируется отдельно).
COPY requirements-pipeline.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-pipeline.txt -r requirements-ml.txt

# Пакет + experiments.wrappers: Perch2Adapter._infer делегирует serving-вызов в
# experiments.wrappers.perch_v2._infer, поэтому этот модуль ОБЯЗАН быть в образе
# (иначе реальная классификация упадёт ImportError). Импорт experiments TF-free.
COPY faun/ ./faun/
COPY experiments/__init__.py ./experiments/__init__.py
COPY experiments/wrappers/ ./experiments/wrappers/

# Непривилегированный пользователь + каталоги job-ов и модели (тома в рантайме).
# Модель в /models монтируется томом faun-models и принадлежит root, но файлы
# world-readable (uid 10001 их читает) — tf.saved_model.load работает.
RUN useradd --create-home --uid 10001 faun \
    && mkdir -p /data/jobs /models \
    && chown -R faun:faun /app /data
USER faun

EXPOSE 8010

# /healthz TF-free (faun.health — stdlib), отвечает сразу; первый /jobs лениво
# поднимает TF+Perch 2 (~десятки секунд). start-period чуть выше slim-образа.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8010/healthz || exit 1

CMD ["uvicorn", "faun.api:app", "--host", "0.0.0.0", "--port", "8010"]
