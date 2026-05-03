# VPS pre-merge runbook (Block 1) — ✅ ЗАВЕРШЕНО 2026-05-03

**Статус:** выполнено через прямой SSH 2026-05-03 ~17:10 MSK. Ниже сохранено для истории / следующих deploy'ев.

**Применённые изменения:**
- `~/apps/faun/.env`: добавлены `FAUN_API_KEY` (48-byte URL-safe random) и `FAUN_FRONTEND_ORIGINS=https://faun.antopkin.ru`. Backup в `.env.bak.1777828062`.
- `client_max_body_size`: **уже было 25M** в `/etc/nginx/nginx.conf:271` для server-блока faun.antopkin.ru — изменения не потребовались (изначальный план 12M был перестраховкой).
- `docker compose -p faun up -d cloud` — restart применил env.

**Verified prod state:**
- `POST /api/v1/demo` без key → 403 (был 503)
- `POST /api/v1/demo` + Origin/Sec-Fetch-Site → 200 (frontend bypass работает)
- `GET /api/v1/mics` → 200 redacted (5 keys: lat/lon/mic_uid/online/zone_type)
- `GET /api/v1/mics` + same-origin → 200 full (9 keys: + battery_pct/installed_at/status/sub_district)
- `GET /api/v1/rangers` без key → 403

---

**Когда выполнять (изначальная инструкция):** перед merge PR-A (#6). Без этого после deploy mutation API на проде продолжит возвращать 503.

**Где выполнять:** на VPS (host из GitHub Secret `VPS_HOST`, у меня — `delphi-press` если alias настроен; иначе `<ip> -p <VPS_PORT>` через ключ из `VPS_SSH_KEY`).

## 1. Добавить env-vars в .env (одна SSH-сессия)

```bash
ssh delphi-press   # или ssh <ip> -p <port>
cd ~/apps/faun
umask 077

# Сгенерировать ключ безопасно — значение нигде не echo'ится
KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")

# FAUN_API_KEY — idempotent: replace если есть, append если нет
if grep -q "^FAUN_API_KEY=" .env; then
  sed -i "s|^FAUN_API_KEY=.*|FAUN_API_KEY=$KEY|" .env
else
  echo "FAUN_API_KEY=$KEY" >> .env
fi

# FAUN_FRONTEND_ORIGINS — same pattern (для same-origin bypass фронта)
if grep -q "^FAUN_FRONTEND_ORIGINS=" .env; then
  sed -i "s|^FAUN_FRONTEND_ORIGINS=.*|FAUN_FRONTEND_ORIGINS=https://faun.antopkin.ru|" .env
else
  echo "FAUN_FRONTEND_ORIGINS=https://faun.antopkin.ru" >> .env
fi

unset KEY
chmod 600 .env

# Restart cloud (не весь стек)
docker compose -p faun up -d cloud

# Verification: ждём 403 (env есть + auth работает), а НЕ 503
sleep 5
curl -s -o /dev/null -w "POST /api/v1/demo (expect 403): %{http_code}\n" \
  -X POST http://127.0.0.1:8002/api/v1/demo \
  -H 'Content-Type: application/json' -d '{}'

curl -s -o /dev/null -w "GET /health (expect 200): %{http_code}\n" \
  http://127.0.0.1:8002/health
```

Если получили `POST → 403` и `GET /health → 200` — env установлен, auth работает.

## 2. Nginx body limit (12MB) — для /api/v1/live/audio + /live/photo

`live/audio` принимает WAV до 10MB + multipart headroom. По умолчанию nginx режет на 1MB → 413 Request Entity Too Large на demo.

```bash
sudo nano ~/apps/delphi-press/nginx/faun-security-headers.conf
```

Добавить в `server { ... }` блок для `faun.antopkin.ru`:
```nginx
client_max_body_size 12m;
```

Reload (зависит от того как nginx запущен):
```bash
# Если nginx как systemd-сервис:
sudo nginx -t && sudo systemctl reload nginx

# Если nginx в docker (delphi-press-nginx-1 — отдельный контейнер):
docker exec delphi-press-nginx-1 nginx -t && docker exec delphi-press-nginx-1 nginx -s reload
```

## 3. Получить ключ для локального демо-тестирования (один раз)

```bash
ssh delphi-press 'awk -F= "/^FAUN_API_KEY=/ {print \$2}" ~/apps/faun/.env' | pbcopy
echo "FAUN_API_KEY скопирован в буфер обмена"
```

Использовать локально:
```bash
export FAUN_API_KEY=$(pbpaste)
python -m demo.presentation_script   # ходит к https://faun.antopkin.ru с X-API-Key
```

## 4. После merge PR-A → проверка прода

```bash
# Mutation должен быть 403 (а НЕ 503!)
curl -s -o /dev/null -w "POST /api/v1/demo (expect 403): %{http_code}\n" \
  -X POST https://faun.antopkin.ru/api/v1/demo \
  -H 'Content-Type: application/json' -d '{}'

# Health всё ещё ОК
curl -s -o /dev/null -w "GET /health (expect 200): %{http_code}\n" \
  https://faun.antopkin.ru/health

# Frontend (открыть в браузере): https://faun.antopkin.ru
# Дашборд должен загрузиться, кнопки «Демо» / «RAG» должны работать (200, не 403)
# Карта микрофонов с battery%/status (same-origin bypass возвращает full payload)
```

## Troubleshooting

| Симптом | Причина | Fix |
|---------|---------|-----|
| `POST → 503` | FAUN_API_KEY env не задан | Повторить шаг 1 |
| `POST → 503` после шага 1 | Контейнер не перечитал .env | `docker compose -p faun restart cloud` |
| Frontend кнопки → 403 | FAUN_FRONTEND_ORIGINS неверный (опечатка в URL) | `grep FAUN_FRONTEND_ORIGINS ~/apps/faun/.env` — должно точно совпадать с тем что в адресной строке браузера, включая `https://` |
| Frontend карта пустая | redacted payload, фронт не падает но без battery% | Bypass не сработал — проверить Origin header в Network tab DevTools |
| Live audio → 413 | nginx body limit не reload | Шаг 2 повторить |
