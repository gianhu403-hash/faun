FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gateway.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements-gateway.txt

COPY . .

EXPOSE 9000

CMD ["python", "-m", "gateway.relay"]
