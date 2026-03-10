# API Reference

## Swagger UI

<swagger-ui src="http://81.85.73.178:8000/openapi.json"/>

---

## Эндпоинты

Все эндпоинты доступны по адресу `http://81.85.73.178:8000`. Auto-generated Swagger UI: `/docs`.

### Core

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/health` | Healthcheck, возвращает `{"status": "ok"}` |
| `GET` | `/` | HTML дашборд (Leaflet карта) |
| `WS` | `/ws` | WebSocket — real-time события |
| `POST` | `/api/v1/demo` | Запуск демо-сценария |
| `POST` | `/demo/start` | Legacy: запуск демо (обратная совместимость) |

### Rangers (Рейнджеры)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/rangers` | Список всех рейнджеров |
| `POST` | `/api/v1/rangers` | Регистрация рейнджера |
| `DELETE` | `/api/v1/rangers/{chat_id}` | Удаление рейнджера |
| `PATCH` | `/api/v1/rangers/{chat_id}/zone` | Обновление зоны мониторинга |
| `PATCH` | `/api/v1/rangers/{chat_id}/active` | Включение/отключение алертов |

#### POST /api/v1/rangers

```json
{
  "name": "Иванов Иван Иванович",
  "chat_id": 123456789,
  "zone_lat_min": 57.05,
  "zone_lat_max": 57.55,
  "zone_lon_min": 44.60,
  "zone_lon_max": 45.40
}
```

### Permits (Разрешения на рубку)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/permits` | Список всех разрешений |
| `POST` | `/api/v1/permits` | Создание разрешения |
| `DELETE` | `/api/v1/permits/{permit_id}` | Удаление разрешения |
| `POST` | `/api/v1/permits/check` | Проверка разрешения по координатам |

#### POST /api/v1/permits/check

Request:

```json
{"lat": 57.30, "lon": 44.80}
```

Response:

```json
{
  "has_valid_permit": true,
  "permits": [{"id": 1, "description": "Санитарная рубка"}]
}
```

### Microphones (Микрофоны)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/mics` | Все микрофоны в сети |
| `GET` | `/api/v1/mics/online` | Только онлайн-микрофоны |
| `PATCH` | `/api/v1/mics/{mic_uid}/status` | Обновить статус (online/offline/broken) |
| `PATCH` | `/api/v1/mics/{mic_uid}/battery` | Обновить заряд батареи |

### AI / ML

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/rag-query` | RAG-запрос (File Search + Web Search) |
| `POST` | `/api/v1/classify` | Cloud classification через DataSphere |
| `POST` | `/api/v1/agent/classify` | AI-верификация классификации |

#### POST /api/v1/rag-query

Request:

```json
{
  "question": "Какая ответственность за незаконную рубку?",
  "context": "chainsaw detected at 57.3N 44.7E"
}
```

Response:

```json
{"answer": "Согласно ст. 260 УК РФ..."}
```

#### POST /api/v1/agent/classify

Request:

```json
{
  "audio_class": "chainsaw",
  "confidence": 0.85,
  "lat": 57.30,
  "lon": 44.80,
  "zone_type": "exploitation",
  "ndsi": -0.45
}
```

Response:

```json
{
  "verified_class": "chainsaw",
  "confidence": 0.85,
  "priority": "critical",
  "context_analysis": "Детектирована бензопила...",
  "recommended_action": "Немедленно направить дрон...",
  "permit_status": "none"
}
```

### Analytics (DataLens)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/datalens/incidents` | JSON инцидентов для DataLens |
| `GET` | `/api/v1/datalens/stats` | Агрегированная статистика |
| `GET` | `/api/v1/incidents/export` | Экспорт CSV для DataLens |
| `GET` | `/api/v1/ai-studio-stack` | Список интеграций AI Studio |

#### GET /api/v1/datalens/stats

```json
{
  "total_incidents": 200,
  "by_class": {"chainsaw": 45, "gunshot": 30, ...},
  "by_status": {"resolved": 120, "pending": 40, ...},
  "by_district": {"Мдальское": 25, ...},
  "avg_response_time_min": 12.5,
  "daily_average": 6.7,
  "detection_rate": 60.0,
  "false_alarm_rate": 15.0
}
```

### FGIS-LK (stub)

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/fgis-lk/forest-unit` | Лесной квартал по координатам |
| `GET` | `/api/v1/fgis-lk/permits` | Активные декларации по координатам |
| `POST` | `/api/v1/fgis-lk/violation` | Подача рапорта о нарушении |

### Workflows

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/api/v1/workflow/definition` | 12-шаговый pipeline (JSON) |
| `POST` | `/api/v1/workflow/run` | Запуск pipeline через Yandex Workflows |

### Live (браузер)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/live/audio` | Классификация аудио из микрофона браузера (webm→wav) |
| `POST` | `/api/v1/live/photo` | Классификация фото из камеры браузера |

### Gateway

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/gateway-event` | Приём события от LoRa Gateway для WebSocket broadcast |

---

## WebSocket Events

Подключение: `ws://81.85.73.178:8000/ws`

| Event | Поля | Описание |
|-------|------|----------|
| `mic_active` | `mics[]` | Активные микрофоны |
| `source_point` | `lat, lon, scenario` | Источник звука |
| `onset_check` | `triggered, energy_ratio` | Результат onset detection |
| `audio_classified` | `class, confidence` | Классификация |
| `location_found` | `lat, lon, error_m` | Триангуляция |
| `agent_decision` | `send_drone, priority, reason` | Решение |
| `drone_moving` | `lat, lon` | Позиция дрона |
| `drone_photo` | `drone_b64` | Фото с дрона |
| `vision_classified` | `description, has_human, has_fire, has_felling` | Vision |
| `alert_sent` | `text, priority` | Алерт отправлен |
| `agent_verified` | `priority, context_analysis, recommended_action` | AI верификация |
| `pipeline_end` | `reason` | Pipeline завершён |

---

## Pydantic Models

```python
class DemoRequest(BaseModel):
    scenario: str = "chainsaw"
    source_lat: float | None = None
    source_lon: float | None = None

class RangerCreate(BaseModel):
    name: str
    chat_id: int
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float

class RangerZoneUpdate(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

class PermitCreate(BaseModel):
    zone_lat_min: float
    zone_lat_max: float
    zone_lon_min: float
    zone_lon_max: float
    valid_from: date
    valid_until: date
    description: str = ""

class PermitCheck(BaseModel):
    lat: float
    lon: float

class RagQueryRequest(BaseModel):
    question: str
    context: str = ""

class ClassifyRequest(BaseModel):
    embeddings: list[float]

class ClassifyAgentRequest(BaseModel):
    audio_class: str
    confidence: float
    lat: float
    lon: float
    zone_type: str = "exploitation"
    ndsi: float | None = None

class GatewayEvent(BaseModel):
    event: str
    # extra="allow" — произвольные доп. поля

class MicStatusUpdate(BaseModel):
    status: str  # online, offline, broken

class MicBatteryUpdate(BaseModel):
    battery_pct: float
```
