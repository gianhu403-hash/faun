#!/usr/bin/env bash
#
# batch_label_raw180.sh — батч-разметка архива raw180 на cluster-alex.
#
# НАЗНАЧЕНИЕ. Прогнать Perch (Apache-2.0) + BirdNET (CC BY-NC-SA, non-commercial!)
# по 180 записям с аудиоловушек в /home/oleg/faun-data/raw180 внутри образа
# faun-ml-cpu и записать:
#   * detections.jsonl  — детекции с псевдо-метками обеих моделей
#                         (source=model:perch / model:birdnet, status=pseudo)
#   * embeddings.npz    — один Perch-эмбеддинг (1280) на детекцию
#
# ВАЖНО. Это РУЧНОЙ утренний прогон на кластере (TF/GPU там, не в CI и не локально).
# Локально TF нет — не запускать; локальная проверка контура — pytest-фикстура
# tests/fixtures/traps_mini (см. docs/labeling.md).
#
# ЛИЦЕНЗИЯ. BirdNET = CC BY-NC-SA: ShareAlike "заражает" дообученную голову.
# Псевдо-метки BirdNET идут ТОЛЬКО на приоритизацию эксперту, НИКОГДА в обучение
# коммерческой модели. Гейт зашит в faun.labeling.training_candidates.
#
# Точная инвокация (то, что выполняется ниже):
#   docker run --rm \
#     -v /home/oleg/faun-data/raw180:/data:ro \
#     -v /home/oleg/faun-data/out/raw180:/out \
#     faun-ml-cpu \
#     python -m faun.cli batch-label \
#       --archive /data \
#       --out /out/detections.jsonl \
#       --emb-out /out/embeddings.npz \
#       --embedder perch \
#       --models perch,birdnet
#
# Ожидаемый вывод: /out/detections.jsonl (N строк) + /out/embeddings.npz [N, 1280],
# плюс JSON-сводка batch_label в stdout (counts / n_detections / n_consensus).
#
# ПРИМЕЧАНИЕ. Подкоманда `faun batch-label` разводится оркестратором (см.
# SHARED_FILE_NEEDS). До её появления тот же контур доступен через
# `python -c` ниже (закомментированный фолбэк).

set -euo pipefail

IMAGE="${FAUN_IMAGE:-faun-ml-cpu}"
DATA_DIR="${FAUN_DATA_DIR:-/home/oleg/faun-data/raw180}"
OUT_DIR="${FAUN_OUT_DIR:-/home/oleg/faun-data/out/raw180}"

echo "[1/4] Проверка входных данных: ${DATA_DIR}"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ОШИБКА: каталог ${DATA_DIR} не найден на кластере." >&2
  exit 1
fi

echo "[2/4] Подготовка каталога вывода: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "[3/4] Запуск Perch+BirdNET батч-разметки в образе ${IMAGE} (CPU-only, TF лениво)"
docker run --rm \
  -v "${DATA_DIR}:/data:ro" \
  -v "${OUT_DIR}:/out" \
  "${IMAGE}" \
  python -m faun.cli batch-label \
    --archive /data \
    --out /out/detections.jsonl \
    --emb-out /out/embeddings.npz \
    --embedder perch \
    --models perch,birdnet

# --- Фолбэк, если подкоманда CLI ещё не разведена оркестратором: --------------
# docker run --rm -v "${DATA_DIR}:/data:ro" -v "${OUT_DIR}:/out" "${IMAGE}" \
#   python -c '
# from faun.labeling import batch_label
# from faun.classification import PerchAdapter, BirdNETAdapter
# from faun.embeddings import PerchEmbedder
# s = batch_label("/data",
#                 {"perch": PerchAdapter(), "birdnet": BirdNETAdapter()},
#                 "/out/detections.jsonl",
#                 emb_out="/out/embeddings.npz",
#                 embedder=PerchEmbedder())
# import json; print(json.dumps(s, ensure_ascii=False, indent=2))'

echo "[4/4] Готово. Артефакты:"
echo "  detections.jsonl : ${OUT_DIR}/detections.jsonl"
echo "  embeddings.npz   : ${OUT_DIR}/embeddings.npz  (Perch, 1280-dim, один на детекцию)"
echo
echo "Напоминание: метки model:birdnet — ТОЛЬКО для эксперта, не для обучения."
