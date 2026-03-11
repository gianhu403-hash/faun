# YDB + DataLens полный сетап

## Промпт для Claude в браузере (Yandex Cloud Console)

Скопировать и вставить в Claude, имеющего доступ к Yandex Cloud Console.

---

```
Мне нужно настроить Yandex Cloud инфраструктуру для проекта Faun (AI-мониторинг леса). Folder ID: b1g5lqh1mqg84cabtejb. Проект уже развёрнут на VPS (81.85.73.178:8000).

## Часть 1: YDB Serverless

### 1a. Проверь, есть ли уже YDB база в folder b1g5lqh1mqg84cabtejb
- Если есть → используем её, запиши endpoint и database path
- Если нет → создай новую:
  - Имя: faun-incidents
  - Тип: Serverless (бесплатный tier)
  - Регион: ru-central1

### 1b. Сервисный аккаунт
- Проверь, есть ли SA с ролью ydb.editor в folder
- Если нет → создай SA `faun-ydb-editor`, назначь роль `ydb.editor`
- Создай authorized key (JSON) для этого SA

### 1c. Результат — пришли мне:
```
YDB_ENDPOINT=grpcs://...
YDB_DATABASE=/ru-central1/...
```
И содержимое JSON-ключа сервисного аккаунта (весь файл).

---

## Часть 2: DataLens → реальные данные

У меня уже есть дашборд: https://datalens.yandex/aaamlpcpp7acu
Сейчас он работает на синтетических данных. Нужно переключить на реальный API.

### 2a. Подключение (Connection)
Создай новое подключение типа «API Connector» в DataLens:
- URL: http://81.85.73.178:8000/api/v1/datalens/incidents
- Метод: GET
- Формат: JSON
- Имя подключения: faun-api-incidents

API возвращает JSON-массив объектов с полями:
- id (string) — UUID инцидента
- timestamp (string) — "YYYY-MM-DD HH:MM:SS"
- lat, lon (float) — координаты
- audio_class (string) — chainsaw, gunshot, engine, axe, fire, background
- confidence (float) — 0.0-1.0
- gating_level (string) — alert, verify, log
- status (string) — pending, accepted, on_site, resolved, false_alarm
- district (string) — участковое лесничество
- response_time_min (float|null) — время отклика в минутах
- ranger_name (string) — имя инспектора
- resolution_details (string) — детали резолюции

Также есть endpoint со статистикой: http://81.85.73.178:8000/api/v1/datalens/stats

### 2b. Датасет
Создай датасет на базе подключения faun-api-incidents. Настрой типы полей:
- timestamp → Дата/Время
- lat, lon, confidence, response_time_min → Число (дробное)
- Остальные → Строка

### 2c. Переподключи существующие чарты дашборда к новому датасету
Если проще — создай новые чарты и замени их на дашборде.

---

## Часть 3: Чарты для дашборда

Создай или обнови чарты на дашборде https://datalens.yandex/aaamlpcpp7acu:

1. **Карта инцидентов** (Геослой / Точки)
   - Координаты: lat, lon
   - Цвет по audio_class
   - Размер по confidence
   - Тултип: timestamp, audio_class, status, ranger_name

2. **Распределение по типам** (Столбчатая диаграмма)
   - X: audio_class
   - Y: COUNT()
   - Цвет по audio_class

3. **Статусы** (Круговая диаграмма)
   - Секции: status
   - Значение: COUNT()

4. **Время отклика по районам** (Столбчатая)
   - X: district
   - Y: AVG(response_time_min)
   - Только resolved

5. **Динамика за 30 дней** (Линейная)
   - X: timestamp (по дням)
   - Y: COUNT()
   - Цвет по audio_class

6. **Селекторы** — добавь на дашборд:
   - Фильтр по audio_class
   - Фильтр по status
   - Фильтр по district
   - Фильтр по дате (timestamp)

---

## Важно

- URL-фильтрация через параметры дашборда НЕ работает (я проверял). Используй селекторы внутри дашборда.
- Для iframe-интеграции: <iframe src="https://datalens.yandex/aaamlpcpp7acu" width="100%" height="800"></iframe>
- Если нужна программная фильтрация — собирай state-хеши для нужных комбинаций (формат ?state=HASH).

Начни с Части 1 (YDB), потом Часть 2 (DataLens подключение), потом Часть 3 (чарты).
```

---

## После получения credentials от Claude в браузере

### 1. SA-ключ → VPS

> Файл `sa-key.json` в корне проекта автоматически доступен в контейнере
> как `/app/sa-key.json` благодаря volume mount `.:/app`.

```bash
# На VPS:
cat > /var/www/ya_hve/sa-key.json << 'KEYEOF'
{
  ... ВСТАВИТЬ JSON-КЛЮЧ ...
}
KEYEOF
chmod 600 /var/www/ya_hve/sa-key.json
```

### 2. Обновить .env на VPS

```bash
# Добавить в /var/www/ya_hve/.env:
YDB_ENDPOINT=grpcs://...
YDB_DATABASE=/ru-central1/...
YDB_SA_KEY_FILE=/app/sa-key.json
```

### 3. Перезапуск

```bash
cd /var/www/ya_hve
docker compose up -d --build
docker compose logs -f cloud --tail=50  # проверить подключение к YDB
```

### 4. Проверка

```bash
# Проверить что API отдаёт данные:
curl -s http://81.85.73.178:8000/api/v1/datalens/incidents | python3 -m json.tool | head -30
curl -s http://81.85.73.178:8000/api/v1/datalens/stats | python3 -m json.tool
```
