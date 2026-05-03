FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-edge.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements-edge.txt

COPY . .

RUN mkdir -p demo/audio demo/photos /tmp/yamnet_cache

EXPOSE 8001

# Create unprivileged user. UID 1000 = matches typical host user (deploy).
# bind-mount in docker-compose preserves host UID, so this lets faun write to bind-mounted dirs.
RUN groupadd -r -g 1000 faun \
    && useradd -r -u 1000 -g faun -d /app -s /sbin/nologin faun \
    && mkdir -p /tmp/yamnet_cache \
    && chown -R faun:faun /app /tmp/yamnet_cache

USER faun

CMD ["sh", "-c", "python demo/generate_audio.py && python -m edge.server"]
