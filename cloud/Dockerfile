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

CMD ["uvicorn", "cloud.interface.main:app", "--host", "0.0.0.0", "--port", "8000"]
