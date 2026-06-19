# ADR 0002 — Single owner for audio preprocessing (`faun/audio.py`)

- Status: Proposed
- Date: 2026-06-19
- Deciders: Faun v2 pipeline team
- Supersedes: —

## Context

Аудио-препроцессинг для входа моделей — три шага: **downmix** в mono, **resample**
(через `soxr` с линейным фолбэком, когда `soxr` недоступен) и **fit_window**
(pad нулями справа или truncate до ровно N сэмплов). Сейчас эта логика **продублирована**
в четырёх местах, попарно почти идентичных:

1. `faun.embeddings._downmix` / `_resample` / `_fit_window` (`faun/embeddings.py`) —
   три module-level функции, тестируемые TF-free.
2. `faun.segmentation.SegmentExtractor._downmix` / `_resample`
   (`faun/segmentation/__init__.py`) — почти посимвольная копия первых двух (resample
   жёстко завязан на `TARGET_SR = 16000`).
3. `faun.classification.perch_v2.Perch2Adapter._prepare` (`faun/classification/perch_v2.py`) —
   inline downmix (`wav.mean(axis=1)`), `soxr.resample`, pad/crop до 160000, **плюс**
   модель-специфичный peak-normalize к `PERCH_V2_PEAK = 0.25`.
4. Адаптеры `perch` / `yamnet` — свои inline-варианты downmix+resample поверх тех же
   wrapper'ов.

Это четыре копии одного алгоритма. Баг в ресэмплинге или в обработке многоканального
сигнала надо чинить в четырёх местах; они уже слегка разъехались (фолбэк без `soxr` в
`embeddings._resample` — линейная интерполяция, в `segmentation._resample` — целочисленная
децимация при кратных частотах, иначе интерполяция).

Дополнительная связка, которую нельзя сломать: `faun/training/dataset.py:25` импортирует
`from faun.embeddings import _downmix, _fit_window, _resample`. Это **замороженный путь
импорта** — `iNatTorchDataset` строит вход трансформера именно через эти три имени.

## Decision

Ввести `faun/audio.py` как **единственного владельца** препроцессинга аудио. Публичные
функции: `downmix(waveform)`, `resample(mono, sr, target_sr)`, `fit_window(mono, win_samples)`.
Семантика — ровно текущая (downmix = среднее по каналам в float32; resample = `soxr` с
линейным фолбэком; fit_window = pad/truncate). Все вызывающие делегируют сюда.

Правила миграции:

1. **`faun.embeddings` СОХРАНЯЕТ module-level имена `_downmix` / `_resample` / `_fit_window`**
   как тонкие ре-экспорты из `faun.audio`. Это нужно, потому что `faun/training/dataset.py`
   импортирует их по этим именам — замороженный контракт. **Инвариант:**
   `faun.embeddings._downmix is faun.audio.downmix` (и так же для двух остальных). Путь
   импорта переживает рефакторинг без правки `dataset.py`.

2. `faun.segmentation.SegmentExtractor._downmix` / `_resample` становятся вызовами
   `faun.audio.downmix` / `faun.audio.resample(mono, sr, TARGET_SR)`. Поведение
   (включая целочисленную децимацию для кратных частот) сохраняется внутри `faun.audio`.

3. **`perch_v2` сохраняет свой peak-normalize** — это модель-специфичный шаг (нормировка к
   `PERCH_V2_PEAK = 0.25`), он остаётся в `Perch2Adapter._prepare` поверх общих
   downmix/resample/fit_window. `faun.audio` нормировку не делает; адаптер навешивает её
   слоем сверху после `fit_window`.

4. Адаптеры `perch`/`yamnet` так же делегируют downmix/resample в `faun.audio`.

Ленивость TF не затрагивается: `faun.audio` — чистый numpy + опциональный `soxr`, как
сейчас в `embeddings`/`segmentation`. Тяжёлые модели остаются в wrapper'ах.

## Consequences

Положительно:

- Один файл, в котором чинится ресэмплинг, downmix и обработка многоканального сигнала, —
  четыре копии схлопываются в одну.
- Расхождение фолбэков без `soxr` (линейная интерполяция vs целочисленная децимация)
  фиксируется в одном месте, под общим тестом.
- Замороженный путь импорта `faun.embeddings._downmix/_resample/_fit_window` выживает —
  `faun/training/dataset.py` не трогаем; инвариант `is`-тождества покрывается тестом.
- Модель-специфичная семантика `perch_v2` (peak-normalize) не меняется — она остаётся
  слоем поверх общего ядра, а не растворяется в нём.

Отрицательно / издержки:

- Ещё один модуль и слой ре-экспортов в `embeddings` (тонкий, но это видимый indirection).
- Тонкие ре-экспорты в `embeddings` — потенциальная ловушка: кто-то может «упростить» их,
  убрав, и сломать `dataset.py`. Защита — тест на инвариант `is`-тождества.

## References

- `faun/embeddings.py` — `_downmix` / `_resample` / `_fit_window` (источник истины сейчас).
- `faun/segmentation/__init__.py` — `SegmentExtractor._downmix` / `_resample` (дубль).
- `faun/classification/perch_v2.py` — `Perch2Adapter._prepare` (+ `PERCH_V2_PEAK` нормировка).
- `faun/training/dataset.py:25` — `from faun.embeddings import _downmix, _fit_window, _resample`
  (замороженный путь импорта, который обязан выжить).
