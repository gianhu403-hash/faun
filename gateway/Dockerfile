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

# Create unprivileged user. UID 1000 = matches typical host user (deploy).
# Gateway has no bind-mount, but uniform UID across services simplifies ops.
RUN groupadd -r -g 1000 faun \
    && useradd -r -u 1000 -g faun -d /app -s /sbin/nologin faun \
    && chown -R faun:faun /app

USER faun

CMD ["python", "-m", "gateway.relay"]
