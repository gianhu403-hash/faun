#!/usr/bin/env bash
# =============================================================================
# finetune_inatsounds.sh — РЕАЛЬНЫЙ fine-tune аудио-трансформера на iNatSounds
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Дообучение САМОГО трансформера (PaSST по умолчанию; AST/BEATs подключаемы) на
#   iNatSounds с размороженным бэкбоном — в отличие от scripts/train_inatsounds.sh,
#   который тренирует ЗАМОРОЖЕННУЮ пробу поверх эмбеддингов. Это разные контуры:
#       train_inatsounds.sh    : embed (frozen) -> logistic probe   (faun.retraining)
#       finetune_inatsounds.sh : raw waveform -> transformer + head (faun.training)  <- ЭТОТ
#
# ГДЕ ЗАПУСКАТЬ
#   cluster-alex, образ faun-ml-torch (GPU, PyTorch-стек; TF/JAX тут не нужны).
#   RTX 2060 SUPER 8GB => тактики на 8GB ниже. НЕ запускать локально (нет GPU,
#   нет hear21passt, нет датасета). Это run-ready handoff — реального прогона
#   здесь НЕ было.
#
# ПРЕДВАРИТЕЛЬНО (ручные шаги — раз)
#   1) Поставить train-зависимости поверх pipeline-стека:
#        pip install -r requirements-pipeline.txt -r requirements-train.txt
#   2) Разложить iNatSounds в дерево root/<species>/<clip> (как
#      tests/fixtures/inatsounds_mini/README):
#        /home/oleg/faun-data/datasets/inatsounds/<species>/<audiofile>
#      (на кластере нет HF/Kaggle-кредов — кладётся вручную).
#
# ИНВОКАЦИЯ (на кластере, внутри faun-ml-torch)
#   bash scripts/finetune_inatsounds.sh \
#       /home/oleg/faun-data/datasets/inatsounds \
#       passt \
#       /home/oleg/faun-data/models/inat_finetune
#   аргументы: <dataset_root> [backbone=passt|ast] [out_checkpoint_dir]
#
# ОЖИДАЕМЫЙ ВЫВОД (stdout, последние строки)
#   dataset: N clips, K species
#   finetune: backbone=passt feature_dim=768 sr=32000 clip_sec=10
#   epoch ...: train_loss=... val_loss=...   (freeze первые --freeze-epochs)
#   early-stop at epoch E (patience=P)        (если сработал)
#   best_epoch=B best_val_loss=...
#   checkpoint -> <out_checkpoint_dir>/  (meta.json + weights.pt)
#   provenance: real-finetune (cluster iNatSounds)
#   (любое число здесь — РЕАЛЬНАЯ species-метрика, т.к. прогон на настоящем
#    iNatSounds с реальным трансформером; синтетика помечается иначе)
#
# 8GB ТАКТИКИ (зашиты в дефолты ниже, правь под память)
#   batch_size 4-8 + grad_accum (эффективный батч = bs*accum), AMP fp16 autocast,
#   заморозка бэкбона первые N эпох -> разморозка, param-group LR (голова > бэкбон
#   ~10x), early-stop по val-loss, CosineAnnealing внутри оптимизатора.
# =============================================================================
set -euo pipefail

DATASET_ROOT="${1:?usage: finetune_inatsounds.sh <dataset_root> [backbone] [out_dir]}"
BACKBONE="${2:-passt}"
OUT_DIR="${3:-/home/oleg/faun-data/models/inat_finetune}"

# --- sanity: датасет существует и непустой -----------------------------------
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root not found: $DATASET_ROOT" >&2
  echo "  Разложи iNatSounds в root/<species>/<clip> (см. README фикстуры)." >&2
  exit 2
fi
if ! find "$DATASET_ROOT" -mindepth 2 -type f \
     \( -iname '*.wav' -o -iname '*.ogg' -o -iname '*.mp3' -o -iname '*.flac' \) \
     | head -1 | grep -q .; then
  echo "ERROR: no audio clips under $DATASET_ROOT/<species>/" >&2
  exit 2
fi

echo "dataset_root : $DATASET_ROOT"
echo "backbone     : $BACKBONE"
echo "out_dir      : $OUT_DIR"

# --- запуск fine-tune через CLI (faun.training.finetune) ----------------------
# 8GB-дефолты: bs=8, grad_accum=4 (эфф. батч 32), freeze 3 эпохи, patience 4.
python -m faun.cli finetune \
  --dataset "$DATASET_ROOT" \
  --model "$BACKBONE" \
  --out "$OUT_DIR" \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 3e-4 \
  --freeze-epochs 3 \
  --patience 4 \
  --device auto \
  --amp \
  --class-weight \
  --seed 42

echo "done -> $OUT_DIR (meta.json + weights.pt)"
echo "ВНИМАНИЕ: species-метрика реальна ТОЛЬКО для этого кластерного прогона."
