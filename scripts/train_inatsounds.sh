#!/usr/bin/env bash
# =============================================================================
# train_inatsounds.sh — обучение видовой пробы на iNatSounds (ТОЛЬКО на кластере)
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Первый прогон, дающий НАСТОЯЩУЮ species-level метрику: на iNatSounds есть
#   ground-truth виды (имя папки = вид). Контур:
#       locate iNatSounds -> embed (Perch/YAMNet) -> train_probe_cv -> species_eval(synthetic=False)
#
# ГДЕ ЗАПУСКАТЬ
#   cluster-alex, образ faun-ml-cpu (TF/JAX CPU-only под cu130). НЕ запускать
#   локально (нет TF, нет датасета). Это run-ready handoff: датасет кладётся
#   ВРУЧНУЮ (на кластере нет HF/Kaggle-кредов).
#
# ПРЕДВАРИТЕЛЬНО (ручной шаг — раз)
#   Скачать iNatSounds где есть креды и разложить в дерево root/<species>/<clip>:
#       /home/oleg/faun-data/datasets/inatsounds/<species>/<audiofile>
#   Раскладка совпадает с tests/fixtures/inatsounds_mini/README.
#
# ИНВОКАЦИЯ (на кластере, внутри faun-ml-cpu)
#   bash scripts/train_inatsounds.sh \
#       /home/oleg/faun-data/datasets/inatsounds \
#       yamnet \
#       /home/oleg/faun-data/models/inat_probe.pkl
#   аргументы: <dataset_root> [embedder=yamnet|perch] [out_probe.pkl]
#
# ОЖИДАЕМЫЙ ВЫВОД (stdout, последние строки)
#   manifest: N clips, K species
#   embeddings: cached -> <emb.npz>
#   train_probe_cv: metric=accuracy value=... ci=[...,...] n=... n_classes=K
#   species_eval(synthetic=False): macro_f1=... provenance=real-eval
#   probe saved -> <out_probe.pkl>
#   (любое число здесь — РЕАЛЬНАЯ species-метрика, т.к. synthetic=False на iNatSounds)
# =============================================================================
set -euo pipefail

DATASET_ROOT="${1:?usage: train_inatsounds.sh <dataset_root> [embedder] [out_probe.pkl]}"
EMBEDDER="${2:-yamnet}"
OUT_PROBE="${3:-/home/oleg/faun-data/models/inat_probe.pkl}"
EMB_CACHE="${EMB_CACHE:-/home/oleg/faun-data/cache/inat_${EMBEDDER}.npz}"
SEED="${SEED:-42}"

echo "[train_inatsounds] dataset_root=${DATASET_ROOT} embedder=${EMBEDDER}"

# -- 0. sanity: датасет на месте -----------------------------------------------
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  echo "       Положите iNatSounds вручную в дерево root/<species>/<clip> (нет HF/Kaggle-кредов на кластере)." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PROBE}")" "$(dirname "${EMB_CACHE}")"

# -- 1..4. embed -> train -> eval -> save (один python-процесс) ----------------
# Тяжёлый TF тянется лениво ВНУТРИ faun.embeddings (Perch/YAMNet wrapper'ы) —
# поэтому это и работает только в faun-ml-cpu на кластере.
DATASET_ROOT="${DATASET_ROOT}" EMBEDDER="${EMBEDDER}" OUT_PROBE="${OUT_PROBE}" \
EMB_CACHE="${EMB_CACHE}" SEED="${SEED}" python - <<'PY'
import os
import numpy as np

from faun.datasets import iNatSoundsDataset
from faun.embeddings import PerchEmbedder, YamnetEmbedder, embed_batch, EmbeddingCache
from faun.retraining import train_probe_cv, species_eval, save_probe

root = os.environ["DATASET_ROOT"]
which = os.environ["EMBEDDER"]
out_probe = os.environ["OUT_PROBE"]
emb_cache = os.environ["EMB_CACHE"]
seed = int(os.environ["SEED"])

# 1. manifest + vocab из дерева root/<species>/<clip>.
ds = iNatSoundsDataset(root)
records = ds.manifest()
vocab = ds.vocab()
print(f"manifest: {len(records)} clips, {len(vocab)} species")

# 2. эмбеддинги (TF тянется лениво внутри wrapper'а; кэш в .npz).
embedder = YamnetEmbedder() if which == "yamnet" else PerchEmbedder()
import soundfile as sf

waveforms = []
species = []
for rec in records:
    wav, sr = sf.read(rec.path)
    waveforms.append((np.asarray(wav, dtype=np.float32), int(sr)))
    species.append(rec.species)

X = embed_batch(waveforms, embedder)            # [N, DIM] (clips, embedder)
y = np.asarray(species)
EmbeddingCache(X, ids=[r.path for r in records], labels=species).save(emb_cache)
print(f"embeddings: cached -> {emb_cache}")

# 3. обучение с CV+CI (переиспользуем общий контур, без дублирования логики).
clf, metrics = train_probe_cv(X, y, seed=seed)
print(
    f"train_probe_cv: metric={metrics['metric']} value={metrics['value']:.4f} "
    f"ci=[{metrics['ci_low']},{metrics['ci_high']}] "
    f"n={metrics['n']} n_classes={metrics['n_classes']}"
)

# 4. РЕАЛЬНАЯ species-метрика: synthetic=False, т.к. это настоящий iNatSounds.
report = species_eval(clf, X, y, synthetic=False)
print(
    f"species_eval(synthetic=False): macro_f1={report['macro_f1']:.4f} "
    f"provenance={report['provenance']}"
)
print("per_species_recall:")
for sp, rec in sorted(report["per_species_recall"].items()):
    print(f"  {sp}: {rec:.3f}")

save_probe(clf, out_probe)
print(f"probe saved -> {out_probe}")
PY

echo "[train_inatsounds] done."
