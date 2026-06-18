# Fine-tuning аудио-трансформера на iNatSounds (`faun.training`)

РЕАЛЬНОЕ дообучение **самого** аудио-трансформера на видовых метках iNatSounds —
в отличие от замороженной пробы поверх эмбеддингов
(`scripts/train_inatsounds.sh` + `faun.retraining.train_probe_cv`). Это два
разных контура; данный — только в `faun.training`.

| контур | вход | что учится | модуль | скрипт |
|---|---|---|---|---|
| frozen probe | эмбеддинги (Perch/YAMNet) | logistic head | `faun.retraining` | `scripts/train_inatsounds.sh` |
| **fine-tune** | raw waveform 32 кГц | трансформер + head | `faun.training` | `scripts/finetune_inatsounds.sh` |

## Где гоняется

Только **cluster-alex**, образ `faun-ml-torch` (GPU, PyTorch-стек). Локально:
торч 2.5.1 CPU есть, но `hear21passt` / iNatSounds / GPU — нет. Поэтому в репо
лежит production-код train-loop + тесты, которые гоняют control-flow на
numpy-стабе (torch-free) + один реальный fwd/bwd под `requires_torch`. Реального
GPU-прогона не было.

Установка train-зависимостей (поверх `requirements-pipeline.txt`):

```
pip install -r requirements-pipeline.txt -r requirements-train.txt
```

## Выбор бэкбона и ЛИЦЕНЗИИ

| backbone | код-лицензия | веса | заметка |
|---|---|---|---|
| **PaSST** (`hear21passt`) | Apache-2.0 | обучены на AudioSet | продуктовый дефолт; оговорка про лицензию датасета AudioSet — проверять перед коммерческим релизом |
| AST (HF `MIT/ast-finetuned-audioset-10-10-0.4593`) | BSD-3 | AudioSet | подключаемая альтернатива (`build_backbone("ast")`) |
| BEATs | код MIT | **кастомная MS-лицензия** | gotcha: веса НЕ MIT — проверить лицензию весов перед использованием |

### PaSST — факты, влияющие на код

- **Нативно 32 кГц** (НЕ 16k — апсемплим апстрим). Датасет
  (`iNatTorchDataset`) ресемплит до 32000 и режет фикс-окно 10 с (≈`input_tdim` 998).
- PaSST гонит **свой mel-фронтенд из raw-waveform** — подаём сырой сигнал
  `[B, sec*32000]`, **не** предрассчитанные мелы.
- `get_basic_model(mode="embed_only")` => **feature_dim = 768**.
  ВНИМАНИЕ: `scene_embedding = 1295 = 527 logits + 768 features` — для головы
  берём **только 768**, не 1295.
- Пересборка головы под N классов: `get_model(arch="passt_s_kd_p16_128_ap486",
  n_classes=N)`.

## Тактики под 8 ГБ (RTX 2060 SUPER)

- `batch_size` 4–8 + **grad-accum** (эффективный батч = `bs * grad_accum`);
- **AMP fp16**: `torch.autocast` + `GradScaler`;
- **freeze -> unfreeze**: заморозить бэкбон первые `freeze_epochs`, потом
  разморозить (учим голову на стабильных фичах, затем тюним всё);
- **param-group LR**: голова в 10–100× быстрее бэкбона;
- **early-stop** по val-loss (`patience`); **CosineAnnealing** LR;
- `num_workers` умеренно (CPU/RAM кластера).

## Архитектура тестируемости (почему control-flow torch-free)

`faun.training.loop.finetune` отделяет **control-flow** от **тензорных операций**:

- control-flow (epoch loop; расписание freeze/unfreeze; счёт микро-шагов для
  grad-accum; история val-loss -> best-epoch + early-stop; запись чекпойнта;
  resume; ветка class-weight) — чистый Python/numpy;
- тензоры (forward/backward/optimizer.step, AMP, CE-loss) — в `_TorchTrainer`
  (torch лениво).

Инжекшн-хуки `finetune(_backbone=..., _loaders=...)` подменяют torch-объекты
numpy-стабом (`_StubBackbone`) и детерминированными лоадерами со
**заскриптованной** последовательностью val-loss. Так ВСЕ перечисленные
поведения наблюдаемы без torch:

- `tests/test_training_loop.py` — 7 control-flow гейтов на стабе + 1 torch fwd/bwd;
- `tests/test_training_dataset.py` — датасет на MINI-фикстуре torch-free + 2 torch-only.

Ни импорт `faun.training`, ни сбор тестов **не** импортируют torch (PEP-562
ленивый ре-экспорт в `__init__.py`).

## Чекпойнт

`save_checkpoint` / `load_checkpoint` бандлят самодостаточный чекпойнт:
`{state_dict, vocab, model_name, feature_dim, sr=32000, clip_sec, epoch,
provenance, extra}`. Метаданные — `meta.json` (torch-free), веса — `weights.pt`
(torch лениво). Инференс восстанавливает голову и маппинг id->вид без знания о
тренировочном коде.

## ЧЕСТНОСТЬ: synthetic vs real

- Любой прогон через стаб/синтетику получает
  `provenance = "SYNTHETIC — not a species metric"`.
- РЕАЛЬНАЯ видовая метрика существует **только** после
  `scripts/finetune_inatsounds.sh` на cluster-alex на настоящем iNatSounds —
  тогда `provenance = "real-finetune (cluster iNatSounds)"`.
- Числа из локальных тестов синтетические и не являются видовой метрикой.
