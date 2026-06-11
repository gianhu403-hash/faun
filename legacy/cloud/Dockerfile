FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    fonts-dejavu-core \
    texlive-luatex \
    texlive-latex-extra \
    texlive-lang-cyrillic \
    fonts-paratype \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements-cloud.txt

RUN python -c "from yandex_ai_studio_sdk import AIStudio; print('SDK OK')"

COPY . .

RUN mkdir -p demo/audio demo/photos /tmp/yamnet_cache

EXPOSE 8000

# Create unprivileged user. UID 1000 = matches typical host user (deploy).
# bind-mount in docker-compose preserves host UID, so this lets faun write to bind-mounted dirs.
RUN groupadd -r -g 1000 faun \
    && useradd -r -u 1000 -g faun -d /app -s /sbin/nologin faun \
    && mkdir -p /tmp/yamnet_cache \
    && chown -R faun:faun /app /tmp/yamnet_cache

USER faun

CMD ["uvicorn", "cloud.interface.main:app", "--host", "0.0.0.0", "--port", "8000"]
