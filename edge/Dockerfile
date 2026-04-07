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

CMD ["sh", "-c", "python demo/generate_audio.py && python -m edge.server"]
