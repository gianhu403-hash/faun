# Тикеты FAUN-1..7 — для review в следующей сессии

**Дата создания:** 2026-05-01 · **Workspace:** consulting · **Project:** Faun (FAUN)
**URL проекта:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/

## Команда (UUIDs)
- Олег Антопкин — `31eeed15-eebb-46b1-bb04-6668a0041e8a`
- Даниил — `7fa29ee1-9fd3-4c5a-acf5-78128b73c687`
- Глеб — `a579fda9-7d60-4da8-bf11-3f47003bfdc5`

## Label IDs (полная таксономия — в `tasks/migration-labels-2026-05-01.md`)
27 labels в Faun проекте: 7 type · 9 area · 4 effort · 3 risk · 4 severity.

## State IDs
- Backlog (default) — `b139a8bb-fcd4-479e-8ebc-430d24b67464`

---

# Созданные тикеты (FAUN-1 .. FAUN-6)

## FAUN-1 [DOC] Структура презы для звонка CFTS — черновик v0.1

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/0dd4af00-e0aa-41e2-8aaa-5f3a1e055613
- **Status:** Backlog
- **Priority:** high
- **Assignee:** unassigned
- **Labels:** `type:content` · `area:docs` · `severity:major` · `effort:S`
- **Target date:** —
- **Тип:** reference-документ (не task), помечен `[DOC]`. Plane CE Pages API не отвечает (404), поэтому положен как work item.
- **Локальный backup:** `~/sandbox/faun/tasks/cfts-call-2026-05-05/preza-structure-v0.1.md`

**Содержит:**
- Контекст звонка (Алёна, CFTS, 30 мин, вторник 5 мая)
- Compressed deck plan (9–10 слайдов из 15)
- Что выкидываем из презы и почему (8 слайдов)
- Главные риски (8 пунктов)
- Highest-leverage interventions
- Open questions (Open-source лицензия / ИП / Кстово-Восточный контакты / Артём)
- Следующие шаги по дням

---

## FAUN-2 Debug cache bug — 100 GB/сутки на VPS, блокер для live-демо

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/021c05b7-9ada-4d43-9aad-ebc2056b63ae
- **Status:** Backlog
- **Priority:** urgent
- **Assignee:** Олег Антопкин
- **Labels:** `type:bug` · `area:deploy` · `severity:blocker` · `effort:M` · `risk:high`
- **Target date:** 2026-05-03 (вс 12:00 — финальный deadline)

**Description:**

### Проблема
На VPS `delphi-press` кэш faun набивается на ~100 GB/сутки. На текущем тренде диск зальётся за 2–3 дня → `faun.antopkin.ru` упадёт. Это блокер для live-демо на звонке с CFTS во вторник 5 мая.

### Симптомы
- Размер кэша растёт линейно ~100 GB/24h
- Не атака на сервер (Олег проверил — другие 5–6 сервисов на VPS работают штатно)
- Воспроизводится на текущем prod-стенде

### Hypotheses (по приоритету)
1. **TF auto-demo OOM-loop** (см. `tasks/lessons.md` #9): auto-demo триггерит TF model load на малой RAM → OOM → restart → re-load. *Mitigation:* `DISABLE_AUTO_DEMO=1` + проверить TF model cache cleanup.
2. **Container logs не rotated** — Docker default без `max-size`. *Mitigation:* log-rotation в `docker-compose.yml`.
3. **SQLite WAL накапливается** — `*-wal` файлы могут расти на active write.
4. **Prometheus / DataLens metrics** копят raw data без TTL.
5. **Demo audio cache** (`yamnet_cache` volume) не очищается.

### Acceptance Criteria
- [ ] Root cause идентифицирован (не догадка — конкретный path / файл / процесс)
- [ ] Fix применён на VPS, проверен 12+ часов без роста кэша
- [ ] Cleanup cron поставлен (если применимо)
- [ ] Size cap на проблемный path / volume (защита от регрессии)
- [ ] Документировано в `tasks/lessons.md` (наследник lesson #9)

### Files / patterns
- `cloud/interface/main.py:_run_demo` (~400)
- `cloud/interface/main.py:_auto_demo`
- `docker-compose.yml` (log-driver config)
- `edge/audio/classifier.py` (`yamnet_cache`)
- `tasks/lessons.md` (#9, #11)

### Deadline
**Суббота 2026-05-03, 18:00** — go/no-go check. Если no-go: Олег + Глеб закрывают вместе в воскресенье до 12:00. Иначе freeze смены кода и фокус на «honest narrative» на звонке.

### Verify
1. `du -sh ~/apps/faun` — стабильно после 12+ часов
2. `df -h` на VPS показывает не растущий disk usage
3. Live-demo end-to-end работает (микрофон → классификация → telegram)

---

## FAUN-3 Пересчитать ML-метрики из confusion matrix slide 8 (accuracy / F1 / per-class P/R)

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/a5072b1c-498f-49af-bbc6-80118d09bd51
- **Status:** Backlog
- **Priority:** high
- **Assignee:** Олег Антопкин
- **Labels:** `type:research` · `area:ml` · `severity:major` · `effort:XS` · `risk:med`
- **Target date:** 2026-05-02 (суббота вечер)

**Description:**

### Зачем
Slide 8 (YAMNet+TDOA) показывает confusion matrix с абсолютными counts (chainsaw 63, gunshot 79, engine 70, axe 54, fire 64, background 222), но без accuracy / precision / recall / F1. Это ловушка #1 в текущей презе: любой ML-grade человек на стороне CFTS (особенно Хохлунов — техарь) поймает за 5 секунд: «а у вас accuracy?». Без готового ответа — credibility пропадает. 30 минут работы → исчезает риск.

### Confusion matrix (из slide 8, fine-tuned v7, leak-free, 2048-D)

| True \ Predicted | chainsaw | gunshot | engine | axe | fire | background |
|---|---|---|---|---|---|---|
| chainsaw | 63 | 0 | 2 | 0 | 0 | 7 |
| gunshot | 0 | 79 | 0 | 0 | 0 | ? |
| engine | 2 | 0 | 70 | 0 | 0 | 22 |
| axe | 0 | 0 | 0 | 54 | 15 | 10 |
| fire | 0 | 0 | 0 | 0 | 64 | 30 |
| background | ? | ? | ? | ? | ? | 222 |

*Значения «?» — досчитать из исходной confusion matrix png-картинки на slide 8 или из исходного notebook'а.*

### Что посчитать
1. Overall accuracy = sum(diag) / sum(all)
2. Macro-F1 = mean(F1 по 6 классам)
3. Per-class precision = TP_i / sum(column_i)
4. Per-class recall = TP_i / sum(row_i)
5. Per-class F1 = 2·P·R/(P+R)
6. Worst-performing class (вероятно `axe` — путается с fire 15× и background 10×)

### Что положить на slide 8 (вместо counts)
Replace block «Дообученная голова (.keras)» на:
- macro-F1 = 0.XX (leak-free, 2048-D)
- accuracy = 0.YY (6 классов, ~ZZ samples)
- worst class: `axe` → F1 = 0.WW (путается с fire 15× + background 10×)
- best class: `gunshot` → F1 ≈ 1.00

Параметры (PHAT, β=0.75, sub-pixel) → speaker notes / follow-up PDF.

### Files / patterns
- Confusion matrix: уже на slide 8 PDF
- Возможные исходники цифр: `docs/results/*.csv` или `docs/notebooks/02_yamnet_test.ipynb`

### Acceptance Criteria
- [ ] Все цифры посчитаны через python (не «на глазок»)
- [ ] Готов 1-line ответ для звонка: «macro-F1 0.XX на 6 классов, leak-free, axe — слабое место, готов план дообучения для v8»
- [ ] Slide 8 текст для Лизы готов (1 параграф)
- [ ] (Опционально) classification_report из sklearn для follow-up PDF

### Verify
1. Сумма всех cells confusion matrix = реальный test set size
2. accuracy между 0 и 1, macro-F1 между 0 и 1
3. Готов ответить на: «почему axe слабый класс?» — потому что он короткий, тон близок к падающим веткам в background

---

## FAUN-4 Counterparty research: 9 precedent CFTS + LinkedIn Алёны и Хохлунова

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/02bd1894-8d3e-4864-9c39-482e293e8d8b
- **Status:** Backlog
- **Priority:** high
- **Assignee:** Даниил
- **Labels:** `type:research` · `severity:major` · `effort:M` · `risk:low`
- **Target date:** 2026-05-03 (воскресенье вечер)

**Description:**

### Зачем
Без понимания их Ideal Customer Profile (ICP) и precedent-проектов мы вслепую — не знаем какой формат engagement они дают (grant / credits / co-development / employment), не можем прицельно сформулировать ask, и не сможем «продать» себя изнутри Yandex (Алёна → Хохлунов).

### Что щерстить (9 precedent проектов CFTS)
Со страницы https://yandex.cloud/ru/social-tech/ecology, для каждого: Habr / пресс-релиз / github / лендинг + scope, стадия, реальные результаты + команда (вуз/стартап/НКО).

Список (priority по релевантности к faun):
1. **ИИ для расследования пожаров (МЧС)** — высокая релевантность, потенциальный pivot
2. **Биомониторинг рыб (СПбГУ)** — биомониторинг, vuz-команда
3. **Биоразнообразие Алтая** — биомониторинг
4. **Экомониторинг Байкала (рыбозапас)** — большой scope
5. **ИИ против борщевика** — другой sensor (не звук)
6. **Чистый берег** — НЕ биомониторинг, но scale (50 км × 3 региона)
7. **Эль-Ниньо (ВШЭ ШАД)** — научная команда, аналог
8. **Снежные барсы** — НЕ упоминать в презе по решению Олега, но знать
9. **Пеплопад Камчатки** — другой sensor

### Что щерстить (counterparty individuals)
**Алёна:** должность в Yandex Cloud / CFTS, tenure, какие проекты вела, Habr / Twitter / Telegram. Hypothesis: delivery PM, первичный фильтр.

**Евгений Хохлунов:** email `ehohlunov@yandex-team.ru`, руководитель направления экологии CFTS, background, что публично пишет про экологию / Machine Learning / архитектуру. Hypothesis: техарь, decision-maker.

### Output
1. **counterparty-brief.md** (1–2 страницы): summary CFTS, таблица 9 проектов, Алёна 1 параграф, Хохлунов 1 параграф, 3–5 inferences
2. **5 prepared questions** для звонка
3. **Talking points для slide 9** — под каждый из 5 критериев CFTS одна строка

### Файл локально
`~/sandbox/faun/tasks/cfts-call-2026-05-05/counterparty-brief.md`

### Acceptance Criteria
- [ ] Из 9 проектов CFTS все 9 проанализированы (даже если public info ноль — отметить «no public data»)
- [ ] Алёна и Хохлунов — basic profile
- [ ] `counterparty-brief.md` написан, ≤2 страницы, без воды
- [ ] 5 prepared questions готовы (текстом)
- [ ] Talking points под 5 критериев CFTS — 5 строк, переданы Олегу
- [ ] Брифует Олега и Глеба до конца воскресенья

### Verify
1. После прочтения brief за 5 минут понятно: что CFTS любит, чего избегает, какой формат engagement ожидать
2. На «расскажи про их precedent» Даниил отвечает 30 сек без подсказок
3. 5 prepared questions проходят sanity check от Олега

---

## FAUN-5 Defensive Machine Learning (ML) pack: метрики, leak-free split, failure modes

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/5873c686-1db1-4381-aa03-60dd4c9513b4
- **Status:** Backlog
- **Priority:** high
- **Assignee:** Глеб
- **Labels:** `type:research` · `area:ml` · `area:tests` · `severity:major` · `effort:M` · `risk:med`
- **Target date:** 2026-05-03 (воскресенье)

**Description:**

### Зачем
Подкрепление к slide 8 пересчёту (FAUN-3). Олег пересчитывает counts → accuracy/F1 на самом slide; Глеб делает 1-pager «defensive ML pack» под капотом — для случая когда Хохлунов копнёт глубже:
- А как train/test split? Не было утечки? Failure modes? Сколько samples в каждом классе? Augmentation? Почему YAMNet, не SOTA?

### Что включить
1. **Метрики** (синхронизировано с FAUN-3): macro-F1, accuracy, per-class P/R/F1, ROC AUC если есть
2. **Leak-free split methodology**: как делили (по recording? segment? source?), гарантия отсутствия shared file, размер каждого split
3. **Datasets used**: AudioSet (YAMNet base), что для fine-tune 6 классов, источники реальных записей
4. **Known failure modes (per class)**:
   - chainsaw путается с engine (~2)
   - engine с background (~22)
   - axe слабый, путается с fire (15) + background (10)
   - fire с background (30)
   - background false positives на ??
5. **План по слабостям → v8**: больше данных axe, augmentation, hard negative mining
6. **Tech FAQ — 7 ответов (≤30 сек)**:
   - Q1: Почему YAMNet, не Whisper / Wav2Vec2 / AST / PANNs?
   - Q2: Зачем TDOA триангуляция, GPS не достаточно?
   - Q3: Какой false positive rate в реальных условиях?
   - Q4: Latency edge → cloud → alert?
   - Q5: Энергопотребление микрофонов?
   - Q6: Сколько микрофонов на 1 кластер триангуляции?
   - Q7: Что дальше за 6 классов?

### Output
- `~/sandbox/faun/tasks/cfts-call-2026-05-05/defensive-ml-pack.md` (1–2 страницы)
- `~/sandbox/faun/tasks/cfts-call-2026-05-05/tech-faq.md`

### Acceptance Criteria
- [ ] Метрики синхронизированы с FAUN-3
- [ ] Train/test split methodology с гарантией leak-free
- [ ] Datasets с размерами
- [ ] Failure modes по 6 классам
- [ ] План по слабостям (3–5 пунктов)
- [ ] Tech FAQ из 7 ответов, ≤30 сек каждый
- [ ] Глеб может на холодную ответить любой из 7 без подсказок

### Verify
1. Передать tech-faq.md Олегу → 3 случайных вопроса голосом → Глеб ≤30 сек каждый
2. Метрики в defensive-ml-pack.md совпадают с slide 8
3. Если CFTS попросит «вышлите ML pack» — артефакт готов

---

## FAUN-6 Pitch контент: cover, slide 5 (Trajectory), slide 9 (Ask + критерии CFTS)

- **URL:** https://tasks.antopkin.ru/consulting/projects/93598537-c48b-471e-a61b-7249657a2f06/issues/2c31c804-11bb-4397-ac3a-a0f197be830b
- **Status:** Backlog
- **Priority:** high
- **Assignee:** Олег Антопкин
- **Labels:** `type:content` · `area:docs` · `severity:major` · `effort:M` · `risk:med`
- **Target date:** 2026-05-03 (воскресенье вечер)

**Description:**

### Зачем
Лизе на понедельник нужен готовый контент для 3 новых/переписанных слайдов. Без этого преза не соберётся.

### Cover (replacement slide 1)
Убрать «команда №3 / технический трек». Добавить:
- Title: «ФАВН — AI-система акустического мониторинга нарушений в лесном фонде»
- Подзаголовок: «Презентация для Yandex Cloud Center for Tech Society»
- Дата: 5 мая 2026
- Команда + контакт Олега (email + Telegram)
- Timeline-ленточка апрель-май: 3+ milestone-точки с момента защиты 11 марта (показать momentum, убрать «студенты после хакатона»)

### Slide 5 — Trajectory (replacement старого roadmap)
3-stage trajectory с параллелью к 6-этапному процессу CFTS:

| Этап (faun) | Что делаем | Сроки | Их этап (CFTS) | Что нам нужно |
|---|---|---|---|---|
| Этап 1 — MVP полишинг | Дотянуть прототип, проверить гипотезы, leak-free retrain | Q3 2026 (~3 мес) | этапы 1–3 | Compute-кредиты + ментор по архитектуре MLOps |
| Этап 2 — Региональный пилот | Внедрить в 1 регионе (50 микрофонов в Нижегородской), верифицировать в полях | Q4 2026 – Q1 2027 (~6 мес) | этап 4 | Cloud opex + warm-intro к региону + co-presentation |
| Этап 3 — Масштабирование | Расширение через ФОИВ: Минприроды, Рослесинфорг, МЧС | 2027+ | этапы 5–6 | Brand + PR + (опц.) acquihire |

Под таблицей 1 строка: «после Этапа 3 — расширение в смежные вертикали (биомониторинг, лесные пожары) уже на стэке Yandex Cloud».

### Slide 9 — Ask + матч с критериями CFTS

**Ask:**
- Cloud-кредиты на Этап 1 + Этап 2 (TBD цифра, согласовать с FAUN-7)
- Ментор по архитектуре MLOps
- Warm intro к региональному партнёру (Нижегородское ГКУ или Минприроды области)
- (Опц.) Co-presentation на их event'ах для PR

**Что мы даём CFTS:**
- Open-source faun (лицензия определяется отдельно)
- Регулярные updates по этапам
- Co-authored case study (когда пилот закроется)
- Возможность acquihire / spin-off

**Матч с 5 критериями CFTS:**
- Значимость: 36 млрд ₽/год ущерб от лесных нарушений
- Технологичность: весь стэк уже на Yandex Cloud — onboarding ≈ 0
- Научность: YAMNet leak-free, TDOA с PHAT, hybrid localization
- Практичность: MVP работает, время реакции 3 нед → 45 мин, 3-этапная trajectory
- Масштабируемость: на стэке готово к расширению на смежные вертикали

### Output
- `~/sandbox/faun/tasks/cfts-call-2026-05-05/pitch-content.md`
- Передаётся Лизе в понедельник утром

### Зависимости
- FAUN-7 (финмодель v1) — нужна цифра cloud-кредитов в Ask
- FAUN-4 (counterparty research) — проверить маппинг 5 критериев

### Acceptance Criteria
- [ ] Cover текст готов (≤80 слов)
- [ ] Slide 5 Trajectory таблица + 1 строка
- [ ] Slide 9 Ask блок (3–4 пункта) + Матч с 5 критериями (5 строк)
- [ ] Цифра cloud-кредитов синхронизирована с FAUN-7
- [ ] Передано Лизе в понедельник утром (≤10:00)

### Verify
1. После прочтения cover за 30 сек понятно: продукт, audience, команда, momentum
2. Slide 5 параллель с 6-этапным процессом CFTS бьётся 1-в-1 (Даниил после FAUN-4 проверяет)
3. Каждая строка матча с критериями отвечает на тот критерий, не размазывается

---

# DRAFT (не создан) — FAUN-7 Финмодель v1

**Status:** не создан, ожидает approve. См. полный draft в transcript / в моём предыдущем сообщении.

- **Title:** Финмодель v1 bottom-up: 1 пилот-регион, 3 revenue scenarios, payback period
- **Priority:** high
- **Assignee:** Даниил
- **Labels:** `type:research` · `area:docs` · `severity:major` · `effort:M` · `risk:med`
- **Target date:** 2026-05-03 (воскресенье)

**Краткое содержание (полный — в FAUN-7 draft в чате):**
- Cost side: CAPEX/микрофон × 50 микрофонов + cloud OPEX/год + human-hours
- Revenue side: 3 scenarios (SaaS per-микрофон / state procurement / integration fee)
- Sensitivity ±20% по 3 параметрам (density / FPR / региональный capex)
- Payback period для 3 сценариев
- Cloud-кредиты для Ask блока FAUN-6
- Output: `finmodel-v1.xlsx` + `finmodel-v1.md`

---

# TBD — ещё не оформлены

| # | Кандидат | Owner | Priority | Why |
|---|---|---|---|---|
| 8 | Cold-emails в Нижегородские ГКУ + Рослесинфорг (Кстово-Восточный) | Даниил | high | Защита от ловушки на slide 6 — «у вас есть соглашение с лесничеством?» |
| 9 | Operational readiness checklist (ИП timeline, банк, NDA, MoU) | Даниил | medium | Готовый ответ на «как переводим compute-кредиты» |
| 10 | Compressed deck verstal (9–10 слайдов) + 3 новых слайда + вычитка | Лиза | high | Финальная сборка, понедельник |
| 11 | 20 «противных» вопросов холодная симуляция + репетиция | все | medium | Понедельник вечер |
| 12 | 1-pager PDF для follow-up к Хохлунову | Олег | medium | Отправляется в течение 24h после звонка |

---

# Где ещё материалы лежат

- **Структура презы:** `~/sandbox/faun/tasks/cfts-call-2026-05-05/preza-structure-v0.1.md` (FAUN-1 backup)
- **Inventory от 3 Explore-сабагентов (119 items):** `~/sandbox/faun/tasks/migration-inventory-2026-05-01.md` (отложенная bulk-миграция, в gitignore через `tasks/migration-*.md`)
- **Labels + project + team UUIDs:** `~/sandbox/faun/tasks/migration-labels-2026-05-01.md`
- **Red-team аудит плана и презы:** `~/sandbox/faun/tasks/cfts-call-2026-05-05/red-team-audit-2026-05-01.md` (создан в этой же сессии — см. ниже)
- **PM Binding в проекте:** в `~/sandbox/faun/CLAUDE.md` секция `## PM Binding`
- **MCP Plane подключение:** `~/sandbox/faun/.mcp.json` (server `plane-consulting`) + `.env` (PLANE_API_KEY, chmod 600)
- **Memory:**
  - `~/.claude/projects/-Users-user-sandbox-faun/memory/reference_plane_mcp.md` — про подключение Plane
  - `~/.claude/projects/-Users-user-sandbox-faun/memory/feedback_acronyms.md` — правило расшифровки аббревиатур
