"""faun.training — РЕАЛЬНЫЙ fine-tune аудио-трансформера на iNatSounds.

Это **отдельный** контур от замороженной пробы (``scripts/train_inatsounds.sh``
+ ``faun.retraining.train_probe_cv``): там голова поверх замороженных эмбеддингов,
здесь — дообучение самого трансформера (PaSST / AST / BEATs) на raw-waveform с
размороженным бэкбоном.

ABSOLUTE RULE: на уровне модуля **нет** ``import torch``. torch тянется лениво
внутри тел функций (PEP-562 ``__getattr__``), поэтому импорт ``faun.training`` и
сбор pytest-тестов НЕ импортируют torch. Оркестрационный слой (эпоха-луп,
freeze/unfreeze, grad-accum, early-stop, checkpoint, resume) отделён от
тензорных операций и тестируется TF/torch-free на чистом numpy-стабе.

ЧЕСТНОСТЬ: реальная species-метрика появляется ТОЛЬКО после прогона
``scripts/finetune_inatsounds.sh`` на cluster-alex (GPU, образ faun-ml-torch) на
настоящем iNatSounds. Локально гоняется только control-flow и один tiny
fwd/bwd под ``requires_torch``. Любое число, посчитанное не на этом прогоне,
синтетическое и не является видовой метрикой.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    # dataset
    "iNatTorchDataset",
    "make_loaders",
    # backbones
    "Backbone",
    "SpeciesHead",
    "build_backbone",
    # loop
    "finetune",
    # checkpoint
    "save_checkpoint",
    "load_checkpoint",
]

if TYPE_CHECKING:  # pragma: no cover - только для статической типизации
    from faun.training.backbones import (
        Backbone,
        SpeciesHead,
        build_backbone,
    )
    from faun.training.checkpoint import load_checkpoint, save_checkpoint
    from faun.training.dataset import iNatTorchDataset, make_loaders
    from faun.training.loop import finetune


# PEP-562: ленивый ре-экспорт, чтобы ``import faun.training`` не тянул torch.
# Тяжёлые символы из dataset/backbones (torch внутри) импортируются лишь при
# первом обращении к ним.
_LAZY: dict[str, tuple[str, str]] = {
    "iNatTorchDataset": ("faun.training.dataset", "iNatTorchDataset"),
    "make_loaders": ("faun.training.dataset", "make_loaders"),
    "Backbone": ("faun.training.backbones", "Backbone"),
    "SpeciesHead": ("faun.training.backbones", "SpeciesHead"),
    "build_backbone": ("faun.training.backbones", "build_backbone"),
    "finetune": ("faun.training.loop", "finetune"),
    "save_checkpoint": ("faun.training.checkpoint", "save_checkpoint"),
    "load_checkpoint": ("faun.training.checkpoint", "load_checkpoint"),
}


def __getattr__(name: str) -> Any:
    """Ленивый импорт публичных символов (PEP-562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
