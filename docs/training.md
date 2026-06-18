# Обучение видовой пробы на iNatSounds

**Дата:** 2026-06-18
**Назначение:** как устроены загрузчик iNatSounds (`faun.datasets`) и оценка по
видам (`faun.retraining.species_eval`), и как запустить РЕАЛЬНОЕ обучение на
кластере. Главный тезис — в разделе «Честность».

---

## 1. TL;DR

- iNatSounds («iNaturalist Sounds Dataset») — **первый источник с истинными
  метками видов**: имя папки = вид (`root/<species>/<audiofile>`). В отличие от
  `raw180` (видового ground-truth НЕТ вообще) тут метрику видов можно измерить
  по-настоящему.
- Локально (без TF, без датасета) собран только **контур**: загрузчик + оценка,
  проверенные на MINI-фикстуре и **синтетических** эмбеддингах.
- **Реальное число видовой точности на сегодня НЕ существует.** Оно появится
  только после прогона `scripts/train_inatsounds.sh` на cluster-alex на
  настоящем iNatSounds (там `species_eval(synthetic=False)`).

---

## 2. Загрузчик: `faun.datasets.iNatSoundsDataset`

Чистый stdlib + numpy, без TensorFlow и без тяжёлых зависимостей. Парсит дерево

```
root/
  <species>/          # имя папки = ground-truth вид
    <audiofile>       # .wav .ogg .mp3 .flac (регистр игнорируется)
```

Публичный API (заморожен):

| метод | возвращает | свойства |
|---|---|---|
| `manifest()` | `list[iNatRecord(path, species)]` | покрывает все аудиофайлы дерева, детерминированный порядок |
| `vocab()` | `dict[species -> int]` | имена видов отсортированы, id непрерывные `0..k-1` |
| `split(seed)` | `(train, val)` | стратифицирован по виду, воспроизводим по `seed`, train+val = весь manifest без пересечений |

Режимы отказа:
- нет/не каталог root → `NotADirectoryError` (явный отказ, не тихий пустой
  результат);
- пустая папка вида (без аудио) → пропускается (нет в `vocab`/`manifest`);
- класс из одного примера → целиком уходит в `train`, `split` не падает.

### MINI-фикстура

`tests/fixtures/inatsounds_mini/` — замороженная крошечная копия раскладки для
юнит-тестов (3 вида, один из них single-sample). Точное содержимое и инварианты —
в `tests/fixtures/inatsounds_mini/README`. Реальный датасет имеет ту же
структуру, отличается только масштаб.

---

## 3. Оценка: `faun.retraining.species_eval`

```python
species_eval(clf, X, y, *, synthetic: bool = True) -> dict
```

Чистый numpy/sklearn (TF не импортируется). Считает:

- `per_species_recall` — `recall_score(average=None)` по видам;
- `macro_f1` — `f1_score(average='macro')`;
- `confusion` — `confusion_matrix` (2D, порядок = `labels`);
- `labels` — упорядоченный список классов;
- `n`, `n_classes`;
- CV-оценку `metric`/`value`/`ci_low`/`ci_high`/`note` — **переиспользуя**
  `train_probe_cv` (логика StratifiedKFold + 95% CI не дублируется);
- `provenance` — **маркер происхождения числа** (см. ниже).

Юнит-тесты (`tests/test_species_eval.py`) реально фитят sklearn-пробу на
синтетических кластеризованных данных и прогоняют `species_eval` — это настоящий
ML-путь без TF, а не зелёный skip.

---

## 4. Реальный запуск на кластере

`scripts/train_inatsounds.sh` — staged launcher для cluster-alex, образ
`faun-ml-cpu`. НЕ запускать локально (нет TF, нет датасета).

Контур: `manifest -> embed (Perch/YAMNet) -> train_probe_cv -> species_eval(synthetic=False) -> save_probe`.

```bash
# 0. (раз) положить iNatSounds вручную — на кластере нет HF/Kaggle-кредов:
#    /home/oleg/faun-data/datasets/inatsounds/<species>/<audiofile>
# 1. внутри faun-ml-cpu на cluster-alex:
bash scripts/train_inatsounds.sh \
    /home/oleg/faun-data/datasets/inatsounds \
    yamnet \
    /home/oleg/faun-data/models/inat_probe.pkl
```

Аргументы: `<dataset_root> [embedder=yamnet|perch] [out_probe.pkl]`. TF тянется
лениво внутри `faun.embeddings` (Perch/YAMNet wrapper'ы), поэтому прогон работает
только на кластере. Эмбеддинги кэшируются в `.npz` (`EmbeddingCache`).

Ожидаемый хвост stdout:

```
manifest: N clips, K species
embeddings: cached -> .../inat_yamnet.npz
train_probe_cv: metric=accuracy value=... ci=[...,...] n=N n_classes=K
species_eval(synthetic=False): macro_f1=... provenance=real-eval
probe saved -> .../inat_probe.pkl
```

Замечание про охват: Perch покрывает **только птиц**, iNatSounds — шире (Perch
для птиц, iNatSounds — широкая аудиомодель).

---

## 5. Честность (правило проекта)

Тон и правило — из `experiments/report/METRICS_HONESTY.md` (не редактируется
этим вектором).

- **Любое видовое число, посчитанное на синтетических эмбеддингах, — НЕ видовая
  точность.** В коде это закреплено: `species_eval(..., synthetic=True)`
  проставляет `provenance = "SYNTHETIC — not a species metric"`. Все юнит-тесты
  и MINI-фикстура — синтетические, и числа из них нельзя выдавать за качество
  продукта.
- **Реальное число существует только из одного места:**
  `species_eval(..., synthetic=False)` внутри `scripts/train_inatsounds.sh` на
  настоящем iNatSounds на кластере. До этого прогона видовой точности у нас нет —
  ни одной цифры.
- **Эталонные метки видов создаёт человек либо берутся из ground-truth датасета
  (iNatSounds), не из предсказаний модели.** Инвентаризация предсказаний — это
  не точность.

---

## 6. Контракт размерностей: обучение и оценка ОДНИМ эмбеддером

Проба обучается на эмбеддингах фиксированной размерности и может оцениваться
только на эмбеддингах ТОЙ ЖЕ размерности. Сейчас в репозитории сосуществуют два
несовместимых YAMNet-вектора:

- `faun.classification.yamnet.YAMNetAdapter.embed` → **1024** (mean-pooling) — это
  то, что видит инференс в проде;
- `faun.embeddings.YamnetEmbedder` → **2048** (`concat(mean, max)`) — пуллинг
  экспериментов/`yamnet_probe`.

Скрестив их (обучить пробу на 1024, оценить на 2048), вы получите ошибку.
`species_eval` ловит это явным `ValueError` ещё до `clf.predict` — не молчаливым
падением sklearn. **Правило:** `scripts/train_inatsounds.sh` и `faun eval-species`
должны использовать один и тот же `--embedder`. Perch консистентен: `PerchEmbedder`
и `PerchAdapter` оба 1280.

> **Открытый вопрос ML-лиду:** какой из двух YAMNet-пуллингов канонизировать
> (1024 mean vs 2048 concat). Это продуктовое решение; ночь его не принимала —
> поставила громкий гейт и эту заметку. См. также выбор Perch v1 (1280, TFHub,
> no-auth) vs Perch 2 (Kaggle): размерности там тоже иные.
