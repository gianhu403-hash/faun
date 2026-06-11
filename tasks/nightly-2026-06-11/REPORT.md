# Утренний отчёт — ночная пересборка Faun v2 (2026-06-11)

План: `~/.claude/plans/fancy-snacking-wadler.md` (v3). Все 9 фаз выполнены. **Всё в main, CI зелёный, 180 тестов, рабочее дерево чистое, на origin одна ветка main + тег v1-hackathon.**

## Что смерджено в main (хронологически)

| Коммит/мердж | Содержание |
|---|---|
| `903a2b8` | deploy.yml → workflow_dispatch-заглушка (VPS мёртв; гейт: ни одного deploy-рана) |
| `521b9b7` | **Reorg**: хакатон → `legacy/` (+ тег `v1-hackathon` @ 0ae49b4 = пин демо), REUSE-ядро → `faun/ml/`, скелет `faun/` + StubAdapter + INTERFACES.md, conftest раздвоен, CI переписан (requirements-pipeline, bandit faun, ветка Oleg убрана). Гейт «ничего не удалено»: 0 deletions против тега |
| `2502323` | Бизнес-доки → `docs/business/{smeta,meetings,hackathon,strategy}` (Расчеты, MoM, ФАВН.pdf, Faun-папка) |
| `4aa2039` | **W1 core**: ingest (info.txt + timestamp из имени), ordering (+gap-детект), segmentation (48k→16k→onset), jobs (atomic manifest), output (CSV+sidecar), storage (LocalFS) — 70 тестов |
| `5fc7549` | **W2 api+ui**: FastAPI POST/GET /jobs + results.csv, CLI `faun process`, одностраничный UI (4 браузерные итерации, скриншоты в faun-nightly-artifacts/) |
| `6443700` | **W3 adapters**: BirdNET/YAMNet/Perch за SpeciesClassifier-протоколом, ленивые импорты (import faun.classification не тянет TF) |
| `ba8f62d` | fix(api): шов run_pipeline ↔ реальные API W1 — пойман e2e-смоуком CLI |
| `f8c2561` | **W4 experiments**: раннер (timeout, graceful skip, vram-замер), обёртки birdnet/perch/clap/yamnet, exp_e0–e10 |
| `883de97` | docs: README/pipeline.md/deployment.md переписаны, CLAUDE.md вычитан, mkdocs nav v2 |
| `6663898` | Фиксы финального аудита: атомарная запись манифеста (гонка поллинга), честный multi-trap sidecar, e2e-тест без моков, timeout+лог в yamnet class-map |
| `07bcf22` | Отчёты экспериментов + bench CSV + бэкпорт clap-хотфикса |

## Эксперименты (реальные данные/модели, кластер)

| Эксп | Результат | Комментарий |
|---|---|---|
| **E3** детектор | agreement **0.467** | onset+ndsi: 30/30 окон, CLAP: 14/30. Дешёвый stage-1 чувствителен, CLAP строг. **30 PNG ждут твоего просмотра**: `cluster:/home/oleg/faun-data/results/e3_windows/` |
| **E4** sanity 180 ГБ | 16 файлов (A1–A4 × 4) | BirdNET убедителен по СЗ-фауне: вальдшнеп (1.00, ночь), зяблик, рябинник, чиж, совиные. Галерея: `e4_gallery/` + в репо `experiments/report/e4_gallery.md` |
| **ESC-50 probe** | Perch **0.9962**, YAMNet **0.9969** AUC | chirping_birds vs rest, 5-fold CV, frozen embeddings. Контрастная задача — обе почти идеальны; различающий тест → ff1010 |
| E1/E2/E5 (ff1010) | skip | ff1010 5.8 ГБ качался медленнее ночи (archive.org). Перезапуск тривиален: дождаться zip → unzip → `runner E1 E2 E5` |
| E10 few-shot | skip (ожидаемо) | нет ключа xeno-canto |

**Рекомендация модели:** Perch 2 (Apache 2.0) как продуктовый классификатор; BirdNET — инвентаризация и бенчмарк, но **лицензия CC BY-NC-SA** (не «BY-NC», как считали: ShareAlike заражает дообученные головы) — для коммерческого пилота нужна альтернатива или договорённость. Детектор: onset.py stage-1 + CLAP-верификатор опционально. Полная матрица: `experiments/report/research-report.md`.

## Статус 180 ГБ
**Скачаны полностью: 1655/1655 файлов, 0 ошибок**, NVMe кластера (`faun-data/raw180/`). Реальная структура: **A1–A4 + RECORDER + «аудиоловушки-офис»** (офисная запись = denoise-референс). **A5 нет** — в брифе было A1..A5.

## Сломалось и починено в полёте
1. Шов api↔W1 (CsvWriter контекстный, AudioFileEntry без load()) — пойман CLI-смоуком, починен до мерджа аудита.
2. `clap.py` × transformers 5.x (pooler_output, audios→audio) — починен на кластере, бэкпортирован.
3. Финальный аудит (SHIP-WITH-CHANGES, 0 блокеров): 3 major пофикшены ночью (atomic manifest, multi-trap sidecar, немокнутый e2e-тест).
4. `faun-ml-cpu` без tensorflow_hub — воркэраунд `/data/pylibs`; **к июлю пересобрать образ** (иначе E2/E5 на ff1010 упадут).
5. uv pip вис на TCP-ретрансмитах в docker-сборке — образы переведены на plain pip.

## Заскипано (и что разблокирует)
- **xeno-canto v3 ключ** (бесплатный, регистрация) → видовые эталоны СЗ + соня/выхухоль → E10, видовая accuracy.
- **Kaggle-аккаунт** → Perch 2 (v2, 1536-dim). Без него — только Perch v1 с TFHub.
- **HF-токен** → не критичен (CLAP и так público).
- FSC22/UrbanSound8K/Watkins — нет чистых no-auth ссылок (формы/скрейпинг).

## Открытые вопросы (не блокировали)
1. Домен faun.antopkin.ru: сейчас → замороженное демо v1 (проверено: vhost на anchor корректен, /health 200). Перекидывать ли на v2-pipeline?
2. BirdNET CC BY-NC-SA для коммерческого пилота — Perch 2 или договорённость.
3. Заводим ли xeno-canto/Kaggle аккаунты (см. выше).
4. CI/CD на кластер (self-hosted runner vs ssh-jump) — июль.
5. Версия сметы (120k vs 130–150k) — бизнес.
6. Локальная ветка `docs/mkdocs-technical-documentation` (7d8708c, upstream удалён, НЕ смержена) — оставлена; решить: добить или удалить.
7. Минорные находки аудита → июль: семантика onset frame_index, UI-поллинг без обратной связи при 500, унификация api-джобстора с faun.jobs, наивный CSV-парсер в UI.

## Гейты финала (все зелёные)
`git status` чист · origin = только main · тег v1-hackathon на origin · CI success · 180 passed · ingress 200 · образы 2/2 (GPU виден) · 180 ГБ 1655/1655 · E3+E4+ESC-50 с реальными значениями в `docs/results/bioacoustics_bench.csv`.
