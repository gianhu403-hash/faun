# Тестирование

## Обзор

Проект содержит **30+ тестовых файлов** и `conftest.py` с общими фикстурами, покрывающих все ключевые компоненты системы.

## Запуск тестов

```bash
# Все тесты
pytest tests/ -x -q

# С подробным выводом
pytest tests/ -v --tb=short

# Один модуль
pytest tests/test_classifier.py -v

# Один тест
pytest tests/test_decider.py::test_high_confidence_alert -v
```

---

## Структура тестов

| Файл | Компонент | Описание |
|------|-----------|---------|
| `conftest.py` | Fixtures | Общие фикстуры: YAMNet моки, фабрики результатов, геометрия микрофонов |
| `test_classifier.py` | Edge / Audio | Классификация YAMNet: загрузка модели, предсказание 6 классов |
| `test_classify_api.py` | Edge / HTTP | Edge classify HTTP API (`:8001`) |
| `test_classify_via_edge.py` | Cloud / Edge | Cloud → Edge HTTP classify с retry |
| `test_onset.py` | Edge / Audio | Onset detection: energy-ratio, пороги, edge cases |
| `test_triangulate.py` | Edge / TDOA | TDOA триангуляция: GCC-PHAT, subpixel, дистанция |
| `test_decider.py` | Edge / Decision | Confidence gating: пороги, пермиты, подавление |
| `test_bot_handlers.py` | Cloud / Telegram | Регистрация: /start, имя, номер, зона |
| `test_bot_workflow.py` | Cloud / Telegram | Workflow: принятие инцидента, статусы, кнопки |
| `test_bot_edge_cases.py` | Cloud / Telegram | Edge cases: concurrent accept, errors |
| `test_bot_format_and_handlers.py` | Cloud / Telegram | Форматирование сообщений, хендлеры |
| `test_drone_bot.py` | Cloud / Telegram | Drone Bot: /start, photo → Vision pipeline |
| `test_drone_scenario_photos.py` | Cloud / Demo | Сценарные фото для drone demo |
| `test_incident_persistence.py` | Cloud / DB | Персистентность: CRUD инцидентов, state machine |
| `test_notification_pipeline.py` | Cloud / Notify | Алерты: compose, rate limiting, отправка |
| `test_permits.py` | Cloud / DB | Разрешения на рубку: проверка, валидация |
| `test_rangers.py` | Cloud / DB | Рейнджеры: регистрация, зоны, CRUD |
| `test_ydb_microphones.py` | Cloud / DB | YDB microphone repository |
| `test_protocol_pdf_endpoint.py` | Cloud / PDF | HTTP endpoint для PDF-протоколов |
| `test_protocol_pdf_latex.py` | Cloud / PDF | LaTeX-генерация протоколов |
| `test_protocol_pdf_photos.py` | Cloud / PDF | Фото в PDF-протоколах |
| `test_rag_agent.py` | Cloud / RAG | RAG-агент: SDK + fallback chain |
| `test_vision_classifier.py` | Cloud / Vision | Gemma 3 Vision: classify_photo, safety net |
| `test_dashboard_readability.py` | Cloud / UI | Читаемость дашборда (размеры, шрифты) |
| `test_pipeline_integration.py` | Integration | Полный pipeline: edge → cloud → telegram |
| `test_demo_live_pipeline.py` | E2E | Live demo pipeline: полный цикл с аудио |
| `test_demo_pipeline.py` | E2E | Demo pipeline end-to-end |
| `test_demo_resilience.py` | E2E | Resilience: pipeline_end при ошибках |
| `test_demo_telegram_alert.py` | E2E | Demo → Telegram alert delivery |
| `test_random_location.py` | Cloud / Demo | Random locations в полигоне Варнавино |
| `test_snooze_and_stale.py` | Cloud / Telegram | Snooze + автозакрытие stale инцидентов |
| `test_remove_stats.py` | Cloud / UI | Удаление СТАТ-вкладки (DataLens migration) |

---

## Fixtures (conftest.py)

### Базовые

| Фикстура | Возвращает | Описание |
|-----------|-----------|---------|
| `sample_rate()` | `16000` | Частота дискретизации |
| `mock_yamnet_model()` | `(scores, embeddings, spectrogram)` | Мок YAMNet: embeddings[5, 1024] |
| `mock_head_model()` | `keras.Model` | Мок 6-классовой головы, вход 2181-dim (2048 features + padding) |

### Фабрики

| Фикстура | Создаёт | Параметры |
|-----------|---------|-----------|
| `audio_result_factory()` | `AudioResult` | Класс (chainsaw/gunshot/engine/axe/fire/background), confidence |
| `triangulation_result_factory()` | `TriangulationResult` | lat, lon, error_m |

### Геометрия

| Фикстура | Описание |
|-----------|---------|
| `triangle_mics()` | 3 микрофона — равносторонний треугольник ~100м (Москва, координаты 55.75°N 37.61°E) |

---

## Покрытие по компонентам

```mermaid
pie title Распределение тестов по компонентам
    "Edge - classifier, onset, TDOA" : 4
    "Decision Engine" : 1
    "Telegram Bot (Ranger + Drone)" : 6
    "Database - incidents, rangers, permits, YDB" : 4
    "Protocol PDF" : 3
    "Vision + RAG" : 2
    "Notification Pipeline" : 1
    "Dashboard + UI" : 2
    "Integration, E2E, Demo" : 7
```

| Компонент | Тестовые файлы | Покрытие |
|-----------|---------------|---------|
| YAMNet classifier | `test_classifier.py`, `test_classify_api.py`, `test_classify_via_edge.py` | Загрузка, предсказание, HTTP API, retry |
| Onset detection | `test_onset.py` | Energy-ratio, пороги, пустой вход |
| TDOA triangulation | `test_triangulate.py` | GCC-PHAT, subpixel, дистанция, ошибки |
| Confidence gating | `test_decider.py` | 3 уровня, пермиты, подавление, дифф. пороги |
| Telegram (Ranger) | `test_bot_handlers.py`, `test_bot_workflow.py`, `test_bot_edge_cases.py`, `test_bot_format_and_handlers.py` | Регистрация, зоны, статусы, edge cases |
| Telegram (Drone) | `test_drone_bot.py` | Photo → Vision → alert pipeline |
| DB persistence | `test_incident_persistence.py`, `test_ydb_microphones.py` | CRUD, state machine, YDB |
| Rangers + Permits | `test_rangers.py`, `test_permits.py` | CRUD, зоны, валидация |
| Protocol PDF | `test_protocol_pdf_endpoint.py`, `test_protocol_pdf_latex.py`, `test_protocol_pdf_photos.py` | HTTP, LaTeX, фото |
| Vision + RAG | `test_vision_classifier.py`, `test_rag_agent.py` | classify_photo, safety net, SDK fallback |
| Notifications | `test_notification_pipeline.py`, `test_snooze_and_stale.py` | Compose, cooldown, snooze, auto-cleanup |
| Dashboard | `test_dashboard_readability.py`, `test_remove_stats.py` | UI размеры, DataLens migration |
| Pipeline E2E | `test_pipeline_integration.py`, `test_demo_*` (4 файла) | E2E flow, resilience |

---

## CI/CD

### GitHub Actions

```yaml
# workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
```

### Локальный запуск

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить тесты
pytest tests/ -x -q

# С coverage (если установлен pytest-cov)
pytest tests/ --cov=cloud --cov=edge --cov-report=html
```

---

## Написание новых тестов

Используйте существующие фикстуры из `conftest.py`:

```python
def test_example(audio_result_factory, triangulation_result_factory):
    """Пример теста с фабриками."""
    result = audio_result_factory(label="chainsaw", confidence=0.95)
    location = triangulation_result_factory(lat=55.75, lon=37.61, error_m=15.0)

    assert result.label == "chainsaw"
    assert location.error_m < 50.0
```

Для тестов Telegram-бота используйте моки `python-telegram-bot`:

```python
from unittest.mock import AsyncMock, MagicMock

async def test_bot_command():
    update = MagicMock()
    context = MagicMock()
    update.message.text = "/start"
    # ... test handler
```
