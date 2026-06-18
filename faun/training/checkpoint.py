"""Чекпойнты fine-tune: bundle {state_dict, vocab, model_name, feature_dim, ...}.

Чекпойнт самодостаточен для инференса: помимо весов хранит словарь видов,
имя бэкбона, размерность фич, аудио-конфиг (sr=32000, clip_sec) и provenance
(SYNTHETIC vs real). Это позволяет восстановить голову и маппинг id->вид без
повторного знания о тренировочном коде.

torch импортируется **лениво** внутри тел функций. Метаданные сериализуются
torch-free (через numpy ``.npz`` для метаданных + отдельный ``.pt`` для весов),
поэтому ``save_checkpoint``/``load_checkpoint`` round-trip-ятся в тестах БЕЗ
state_dict (control-flow), а с реальными весами — на кластере.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Версия формата чекпойнта (для будущей миграции).
CHECKPOINT_FORMAT = 2

# sr трансформера (PaSST нативно 32 кГц). Веса/инференс ожидают именно это.
DEFAULT_SR = 32_000


def save_checkpoint(
    path: str | Path,
    *,
    state_dict: Any | None,
    vocab: dict[str, int],
    model_name: str,
    feature_dim: int,
    epoch: int,
    provenance: str,
    sr: int = DEFAULT_SR,
    clip_sec: float = 10.0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Записать чекпойнт (атомарно). Возвращает путь.

    Раскладка: один каталог ``path/`` (или ``path`` трактуется как директория)
    с ``meta.json`` (vocab/model_name/feature_dim/epoch/provenance/sr/clip_sec/
    extra) и опциональным ``weights.pt`` (torch ``state_dict``). Когда
    ``state_dict is None`` — пишутся только метаданные (control-flow тесты).

    torch нужен только если ``state_dict is not None`` (импортируется лениво).
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "format": CHECKPOINT_FORMAT,
        "vocab": dict(vocab),
        "model_name": str(model_name),
        "feature_dim": int(feature_dim),
        "epoch": int(epoch),
        "provenance": str(provenance),
        "sr": int(sr),
        "clip_sec": float(clip_sec),
        "has_weights": state_dict is not None,
        "extra": dict(extra) if extra else {},
    }

    tmp_meta = out / ".meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    tmp_meta.replace(out / "meta.json")

    if state_dict is not None:
        import torch

        tmp_w = out / ".weights.pt.tmp"
        torch.save(state_dict, tmp_w)
        tmp_w.replace(out / "weights.pt")

    return out


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Прочитать чекпойнт -> dict с метаданными и (если есть) ``state_dict``.

    Ключи результата: ``vocab``, ``model_name``, ``feature_dim``, ``epoch``,
    ``provenance``, ``sr``, ``clip_sec``, ``extra``, ``format`` и ``state_dict``
    (``None`` если веса не сохранялись). torch импортируется лениво и только при
    наличии ``weights.pt``.
    """
    src = Path(path)
    meta_path = src / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"checkpoint meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text("utf-8"))

    state_dict: Any | None = None
    weights = src / "weights.pt"
    if weights.is_file():
        import torch

        state_dict = torch.load(weights, map_location=map_location, weights_only=True)

    return {
        "format": meta.get("format", CHECKPOINT_FORMAT),
        "vocab": meta["vocab"],
        "model_name": meta["model_name"],
        "feature_dim": meta["feature_dim"],
        "epoch": meta["epoch"],
        "provenance": meta["provenance"],
        "sr": meta.get("sr", DEFAULT_SR),
        "clip_sec": meta.get("clip_sec", 10.0),
        "extra": meta.get("extra", {}),
        "state_dict": state_dict,
    }
