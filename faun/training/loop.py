"""Тренировочный луп fine-tune — CONTROL FLOW отделён от torch-тензоров.

Ядро тестируемости (gate 2): эпоха-луп, расписание freeze/unfreeze, счёт
микро-шагов для grad-accum, история val-loss для best-epoch + early-stop,
запись чекпойнта, resume и ветка class-weight — всё это ЧИСТЫЙ control-flow.
Тензорные операции (forward/backward/optimizer.step на torch) изолированы в
:class:`_TorchTrainer`. Через инжекшн-хуки ``_backbone`` / ``_loaders`` тесты
подменяют их numpy-стабом + детерминированными лоадерами и наблюдают ВСЕ
перечисленные поведения БЕЗ torch.

ABSOLUTE RULE: на уровне модуля нет ``import torch`` — он только лениво внутри
:class:`_TorchTrainer` (реальный путь, кластер).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from faun.training.checkpoint import load_checkpoint, save_checkpoint

# Маркеры честности: provenance чекпойнта.
_SYNTHETIC_PROVENANCE = "SYNTHETIC — not a species metric"
_REAL_PROVENANCE = "real-finetune (cluster iNatSounds)"


# ---------------------------------------------------------------------------
# Наблюдаемые помощники control-flow (numpy, без torch)
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    """numpy-«оптимизатор» для control-flow тестов: считает ``step``/``zero_grad``.

    Наблюдаемые: ``step_calls`` (раз в ``grad_accum`` микро-батчей) и
    ``zero_grad_calls``.
    """

    def __init__(self) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1


class _StubTrainer:
    """torch-free тренер: forward стаба + детерминированный псевдо-loss.

    Реального backward нет; цель — наблюдаемость control-flow. Каждый микро-батч
    инкрементит ``micro_steps``; ``optimizer_step`` дергается лупом по расписанию
    grad-accum (наблюдается через :class:`_FakeOptimizer`).
    """

    def __init__(self, backbone: Any, optimizer: _FakeOptimizer) -> None:
        self.backbone = backbone
        self.optimizer = optimizer
        self.micro_steps = 0

    def train_micro_batch(self, batch: Any) -> float:
        self.micro_steps += 1
        waveforms, _labels = batch
        feats = self.backbone.forward(waveforms)
        return float(np.mean(np.abs(feats)))

    def optimizer_step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()


def _compute_class_weights(labels: Iterable[int], n_classes: int) -> np.ndarray:
    """Инверсно-частотные (balanced) веса классов -> [n_classes] float32.

    Класс без примеров получает вес 0 (без деления на ноль). Это тело ветки
    ``class_weight=True``; в тестах наблюдается факт вызова + значения.
    """
    arr = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(arr, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    n_present = int(nonzero.sum())
    if n_present and total:
        weights[nonzero] = total / (n_present * counts[nonzero])
    return weights.astype(np.float32)


def _iter_labels(loader: Any) -> list[int]:
    """Собрать все метки из лоадера (для class-weight). torch-free."""
    out: list[int] = []
    for _waveforms, labels in loader:
        if hasattr(labels, "tolist"):
            out.extend(int(x) for x in labels.tolist())
        elif np.ndim(labels) == 0:
            out.append(int(labels))
        else:
            out.extend(int(x) for x in labels)
    return out


# ---------------------------------------------------------------------------
# Реальный torch-тренер (ленивый torch, только кластер)
# ---------------------------------------------------------------------------


class _TorchTrainer:  # pragma: no cover - реальный путь, только под torch/кластер
    """Реальный тренер: forward бэкбона+головы, CE-loss, AMP, backward."""

    def __init__(
        self,
        backbone: Any,
        head: Any,
        optimizer: Any,
        *,
        class_weights: Any | None,
        amp: bool,
        device: Any,
    ) -> None:
        import torch

        self.torch = torch
        self.backbone = backbone
        self.head = head
        self.optimizer = optimizer
        self.device = device
        self.amp = bool(amp) and getattr(device, "type", "cpu") == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        weight = (
            torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights is not None
            else None
        )
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        self.micro_steps = 0

    def train_micro_batch(self, batch: Any) -> float:
        torch = self.torch
        waveforms, labels = batch
        waveforms = waveforms.to(self.device)
        labels = labels.to(self.device)
        self.micro_steps += 1
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            feats = self.backbone.forward(waveforms)
            logits = self.head(feats)
            loss = self.criterion(logits, labels)
        self.scaler.scale(loss).backward()
        return float(loss.detach().cpu().item())

    def optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def state_dict(self) -> Any:
        return self.head.state_dict()


# ---------------------------------------------------------------------------
# Публичный луп (FROZEN сигнатура)
# ---------------------------------------------------------------------------


def finetune(
    dataset_root: str | Path,
    *,
    vocab: dict[str, int] | None = None,
    model: str = "passt",
    out: str | Path,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "auto",
    amp: bool = True,
    grad_accum: int = 2,
    freeze_epochs: int = 3,
    patience: int = 4,
    class_weight: bool = True,
    seed: int = 42,
    resume: str | Path | None = None,
    _backbone: Any | None = None,
    _loaders: tuple[Any, Any] | None = None,
) -> dict:
    """Дообучить аудио-трансформер на iNatSounds; вернуть сводку прогона.

    FROZEN сигнатура. ``_backbone`` / ``_loaders`` — **тест-инжекшн** (подчёркнуты):
    когда переданы, луп использует их вместо построения реальных torch-объектов,
    так control-flow гоняется без torch.

    Control-flow (наблюдаемо на стабе, torch-free):
    * расписание freeze/unfreeze: бэкбон заморожен пока ``epoch < freeze_epochs``,
      размораживается ровно на ``epoch == freeze_epochs``;
    * grad-accum: ``optimizer.step`` раз в ``grad_accum`` микро-батчей;
    * история val-loss -> best-epoch (min) + early-stop (через ``patience`` эпох
      без улучшения);
    * чекпойнт лучшей эпохи; ``resume`` продолжает с сохранённой эпохи;
    * ветка ``class_weight``: веса из распределения меток train-лоадера.

    Возвращает dict: ``epochs_run``, ``best_epoch``, ``best_val_loss``,
    ``val_loss_history``, ``early_stopped``, ``start_epoch``, ``optimizer_steps``,
    ``class_weights`` (или None), ``checkpoint``, ``provenance``, ``n_classes``.
    """
    stub_mode = _backbone is not None or _loaders is not None

    # --- лоадеры -----------------------------------------------------------
    if _loaders is not None:
        train_loader, val_loader = _loaders
    else:  # pragma: no cover - реальный путь (torch + датасет, только кластер)
        from faun.training.dataset import make_loaders

        if vocab is None:
            from faun.datasets import iNatSoundsDataset

            vocab = iNatSoundsDataset(dataset_root).vocab()
        train_loader, val_loader = make_loaders(
            str(dataset_root), vocab, seed=seed, batch_size=batch_size
        )

    if vocab is None:
        # Инжектированный режим без vocab: выводим n_classes из меток лоадера.
        n_classes = (max(_iter_labels(train_loader)) + 1) if not stub_mode else 0
        vocab = {str(i): i for i in range(n_classes)}
    n_classes = len(vocab)

    # --- бэкбон ------------------------------------------------------------
    if _backbone is not None:
        backbone = _backbone
    else:  # pragma: no cover - реальный путь
        from faun.training.backbones import build_backbone

        backbone = build_backbone(model, freeze=True)

    feature_dim = int(getattr(backbone, "feature_dim", 0))

    # --- class-weight ветка -----------------------------------------------
    class_weights: np.ndarray | None = None
    if class_weight:
        class_weights = _compute_class_weights(_iter_labels(train_loader), n_classes)

    # --- тренер + оптимизатор ---------------------------------------------
    if stub_mode:
        optimizer = _FakeOptimizer()
        trainer: Any = _StubTrainer(backbone, optimizer)
        head: Any = None
        provenance = _SYNTHETIC_PROVENANCE
    else:  # pragma: no cover - реальный путь (torch)
        trainer, head, optimizer, provenance = _build_torch_trainer(
            backbone, feature_dim, n_classes, lr, amp, device, class_weights
        )

    # --- resume ------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1
    if resume is not None:
        ckpt = load_checkpoint(resume)
        start_epoch = int(ckpt["epoch"]) + 1
        # Лучшее значение из resume неизвестно точно; берём из extra если есть.
        best_val_loss = float(ckpt.get("extra", {}).get("best_val_loss", "inf"))
        best_epoch = int(ckpt.get("extra", {}).get("best_epoch", -1))

    # Скрипт val-loss для control-flow тестов: атрибут на val_loader.
    scripted_val = getattr(val_loader, "val_loss_script", None)

    val_loss_history: list[float] = []
    epochs_run = 0
    early_stopped = False
    epochs_since_best = 0

    # --- эпоха-луп ---------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # freeze/unfreeze расписание (наблюдаемо через флаг стаба).
        if epoch < freeze_epochs:
            if hasattr(backbone, "freeze") and not getattr(backbone, "frozen", True):
                backbone.freeze()
        else:
            if hasattr(backbone, "unfreeze") and getattr(backbone, "frozen", False):
                backbone.unfreeze()

        # train: микро-батчи + grad-accum.
        micro = 0
        for batch in train_loader:
            trainer.train_micro_batch(batch)
            micro += 1
            if micro % grad_accum == 0:
                trainer.optimizer_step()
        # хвостовой шаг для оставшихся микро-батчей (неполный аккум).
        if micro % grad_accum != 0:
            trainer.optimizer_step()

        # val-loss: скрипт (тесты) либо реальная оценка (кластер).
        if scripted_val is not None:
            val_loss = float(scripted_val[epoch])
        else:  # pragma: no cover - реальная val-оценка под torch
            val_loss = _evaluate(trainer, val_loader)
        val_loss_history.append(val_loss)
        epochs_run += 1

        # best-epoch + чекпойнт лучшей.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            save_checkpoint(
                out,
                state_dict=(None if stub_mode else trainer.state_dict),
                vocab=vocab,
                model_name=model,
                feature_dim=feature_dim,
                epoch=epoch,
                provenance=provenance,
                extra={"best_val_loss": best_val_loss, "best_epoch": best_epoch},
            )
        else:
            epochs_since_best += 1

        # early-stop: ровно через ``patience`` эпох без улучшения.
        if epochs_since_best >= patience:
            early_stopped = True
            break

    return {
        "epochs_run": epochs_run,
        "start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_loss_history": val_loss_history,
        "early_stopped": early_stopped,
        "optimizer_steps": int(getattr(optimizer, "step_calls", -1)),
        "class_weights": (
            class_weights.tolist() if class_weights is not None else None
        ),
        "checkpoint": str(out),
        "provenance": provenance,
        "n_classes": n_classes,
        "feature_dim": feature_dim,
    }


def _build_torch_trainer(
    backbone: Any,
    feature_dim: int,
    n_classes: int,
    lr: float,
    amp: bool,
    device: str,
    class_weights: np.ndarray | None,
):  # pragma: no cover - реальный путь, только под torch/кластер
    """Собрать torch-тренер: голова, param-group LR (голова > бэкбон), AMP."""
    import torch

    from faun.training.backbones import SpeciesHead

    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    head = SpeciesHead(feature_dim, n_classes).to(dev)
    backbone_net = getattr(backbone, "net", None)
    param_groups: list[dict[str, Any]] = [{"params": head.parameters(), "lr": lr}]
    if backbone_net is not None:
        # Бэкбон учится медленнее головы (10x), см. docs/finetuning.md.
        param_groups.append({"params": backbone_net.parameters(), "lr": lr / 10.0})
    optimizer = torch.optim.AdamW(param_groups)
    trainer = _TorchTrainer(
        backbone, head, optimizer, class_weights=class_weights, amp=amp, device=dev
    )
    return trainer, head, optimizer, _REAL_PROVENANCE


def _evaluate(trainer: Any, val_loader: Any) -> float:  # pragma: no cover - torch
    """Средний CE-loss по val-лоадеру (torch, режим eval)."""
    import torch

    trainer.head.eval()
    losses: list[float] = []
    with torch.no_grad():
        for waveforms, labels in val_loader:
            waveforms = waveforms.to(trainer.device)
            labels = labels.to(trainer.device)
            feats = trainer.backbone.forward(waveforms)
            logits = trainer.head(feats)
            losses.append(float(trainer.criterion(logits, labels).item()))
    trainer.head.train()
    return float(np.mean(losses)) if losses else float("inf")
