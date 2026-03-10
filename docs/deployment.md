# Деплой

## Docker Compose

Система запускается через `docker compose` с тремя сервисами:

```yaml
services:
  cloud:       # FastAPI + Telegram-бот + AI-агенты
  edge:        # YAMNet classifier + TDOA + Decision engine
  lora_gateway: # LoRa mesh relay
```

### Сервисы

| Сервис | Порт | Healthcheck | Зависимости |
|--------|------|-------------|-------------|
| **cloud** | `:8000` | `GET /health` (30s interval, 10s timeout, 3 retries) | `lora_gateway` |
| **edge** | — | — | — |
| **lora_gateway** | `:9000` | — | — |

### Volumes

- `.:/app` — монтирование кода в контейнер cloud
- `./demo/audio:/app/demo/audio` — демо-аудио для edge
- `yamnet_cache:/tmp/yamnet_cache` — общий кэш модели YAMNet (shared между cloud и edge)

Все сервисы используют политику `restart: unless-stopped`.

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    libsndfile1 libsndfile1-dev ffmpeg curl \
    fonts-dejavu-core build-essential

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Валидация SDK
RUN python -c "import yandex_ai_studio_sdk; print('SDK OK')"

# Рабочие директории
RUN mkdir -p demo/audio demo/photos /tmp/yamnet_cache

EXPOSE 8000
CMD ["uvicorn", "cloud.interface.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Base image:** Python 3.11-slim
**Системные пакеты:** libsndfile (аудио I/O), ffmpeg (конвертация), curl (healthcheck)

---

## VPS-сервер

| Параметр | Значение |
|----------|---------|
| IP | `81.85.73.178` |
| SSH | `ssh root@81.85.73.178` |
| Код | `/var/www/ya_hve` (ветка `main`) |
| Docker | docker compose v2 (2.34.0) |
| ОС | Ubuntu 22.04 |
| RAM | 1.9 GB |
| Модель | YAMNet v7 загружена и кэширована |

---

## Переменные окружения

### Обязательные

| Переменная | Описание |
|------------|---------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram-бота (`@ya_faun_bot`) |
| `YANDEX_API_KEY` | API-ключ Yandex Cloud |
| `YANDEX_FOLDER_ID` | ID каталога Yandex Cloud (`b1g5lqh1mqg84cabtejb`) |

### AI Studio / RAG

| Переменная | Описание |
|------------|---------|
| `SEARCH_INDEX_ID` | ID индекса File Search (`fvttk7bjvnm39qogtoep`) |
| `DATASPHERE_NODE_ID` | ID ноды DataSphere для обучения |

### YDB Serverless (опционально)

| Переменная | Значение по умолчанию |
|------------|----------------------|
| `YDB_ENDPOINT` | `grpcs://ydb.serverless.yandexcloud.net:2135` |
| `YDB_DATABASE` | `/ru-central1/b1g.../etn...` |
| `YDB_SA_KEY_FILE` | `/app/sa-key.json` |

### Опциональные

| Переменная | По умолчанию | Описание |
|------------|-------------|---------|
| `ALERT_COOLDOWN_SECONDS` | `300` | Кулдаун между алертами одного типа |
| `CLOUD_API_URL` | `http://cloud:8000` | URL cloud-сервиса для edge |
| `LORA_GATEWAY_HOST` | `lora_gateway` | Хост LoRa gateway |
| `LORA_GATEWAY_PORT` | `9000` | Порт LoRa gateway |
| `MIC_MODE` | `sim` | Режим микрофонов (`sim` / `real`) |
| `DEMO_SCENARIO` | `chainsaw` | Демо-сценарий |
| `DISABLE_AUTO_DEMO` | — | Если установлена — отключает auto-demo при старте |

### Координаты микрофонов (Варнавино, fallback)

| Переменная | Значение |
|------------|---------|
| `MIC_A_LAT` / `MIC_A_LON` | `57.3697` / `44.6200` |
| `MIC_B_LAT` / `MIC_B_LON` | `57.3752` / `44.6345` |
| `MIC_C_LAT` / `MIC_C_LON` | `57.3631` / `44.6489` |

Используются только если в БД менее 3 онлайн-микрофонов.

---

## Команды деплоя

```bash
# Первоначальная настройка
ssh root@81.85.73.178
cd /var/www/ya_hve
cp .env.example .env
nano .env  # заполнить секреты

# Сборка и запуск
docker compose build
docker compose up -d

# Проверка состояния
docker compose ps
docker compose logs -f cloud

# Обновление
cd /var/www/ya_hve
git pull origin main
docker compose build
docker compose up -d

# Перезапуск одного сервиса
docker compose restart cloud
```

---

## YDB Setup

Для использования YDB Serverless вместо SQLite:

1. Создать сервисный аккаунт в Yandex Cloud
2. Скачать ключ (`sa-key.json`) и положить в `/app/sa-key.json`
3. Установить переменные `YDB_ENDPOINT`, `YDB_DATABASE`, `YDB_SA_KEY_FILE`
4. Таблицы создаются автоматически при первом запуске (см. [База данных](database.md))

---

## Nginx (документация)

Для хостинга MkDocs-документации через nginx:

```nginx
server {
    listen 8080;
    server_name 81.85.73.178;
    root /var/www/ya_hve/site;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

Сборка: `mkdocs build` создаёт статику в `site/`.
