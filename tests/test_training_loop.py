"""Control-flow тесты тренировочного лупа fine-tune — TF/torch-FREE на стабе.

ABSOLUTE RULE здесь соблюдается: НЕТ ``import torch`` на уровне модуля. Весь
control-flow (epoch loop, freeze/unfreeze, grad-accum, early-stop, best-epoch,
checkpoint, resume, class-weight) гоняется на чистом numpy-стабе через
инжекшн-хуки ``finetune(_backbone=..., _loaders=...)``. Единственный torch-тест
делает ``pytest.importorskip("torch")`` ВНУТРИ тела.

В каждом тесте помечено, что покрыто стабом (control-flow), а что — torch-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from faun.training.backbones import _StubBackbone
from faun.training.checkpoint import load_checkpoint, save_checkpoint
from faun.training.loop import _compute_class_weights, finetune


# ---------------------------------------------------------------------------
# Детерминированные фейковые лоадеры (numpy, torch-free)
# ---------------------------------------------------------------------------


class _FakeLoader:
    """Лоадер детерминированных батчей ``(waveforms[B,win], labels[B])`` (numpy).

    На val-лоадере дополнительно несёт ``val_loss_script`` — заскриптованную
    последовательность val-loss по эпохам, которую читает луп (наблюдаемость
    best-epoch и early-stop без реального forward/backward).
    """

    def __init__(
        self,
        n_batches: int,
        batch_size: int = 2,
        win: int = 8,
        *,
        labels: list[int] | None = None,
        val_loss_script: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self._n = n_batches
        self._bs = batch_size
        self._win = win
        self._labels = labels
        self._seed = seed
        if val_loss_script is not None:
            self.val_loss_script = val_loss_script

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        for b in range(self._n):
            wf = rng.standard_normal((self._bs, self._win)).astype(np.float32)
            if self._labels is not None:
                start = b * self._bs
                labs = np.asarray(
                    self._labels[start : start + self._bs], dtype=np.int64
                )
            else:
                labs = np.zeros(self._bs, dtype=np.int64)
            yield wf, labs


def _run(
    tmp_path,
    *,
    val_script,
    epochs=10,
    grad_accum=2,
    freeze_epochs=3,
    patience=4,
    class_weight=False,
    train_batches=4,
    train_labels=None,
    resume=None,
    backbone=None,
):
    """Хелпер: прогнать ``finetune`` на стабе с заскриптованным val-loss."""
    bb = backbone if backbone is not None else _StubBackbone(feature_dim=8, seed=1)
    train = _FakeLoader(train_batches, labels=train_labels, seed=1)
    val = _FakeLoader(1, val_loss_script=val_script, seed=2)
    vocab = {"a": 0, "b": 1}
    out = tmp_path / "ckpt"
    summary = finetune(
        tmp_path,
        vocab=vocab,
        out=out,
        epochs=epochs,
        grad_accum=grad_accum,
        freeze_epochs=freeze_epochs,
        patience=patience,
        class_weight=class_weight,
        resume=resume,
        _backbone=bb,
        _loaders=(train, val),
    )
    return summary, bb, out


# ---------------------------------------------------------------------------
# 1. early-stop ровно через patience эпох после лучшей
# ---------------------------------------------------------------------------


def test_early_stop_fires_at_patience_after_best(tmp_path):
    """STUB-COVERED (control-flow): early-stop через ``patience`` эпох без улучшения.

    val-loss падает до эпохи 2 (best), затем растёт. patience=4 => стоп ровно на
    epoch 2+4 = 6 (т.е. после 4 эпох без улучшения).
    """
    # epoch:        0    1    2*   3    4    5    6    7 ...
    script = [0.9, 0.7, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    summary, _bb, _out = _run(tmp_path, val_script=script, patience=4, epochs=10)

    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == pytest.approx(0.5)
    assert summary["early_stopped"] is True
    # эпохи 0..6 включительно => 7 прогонов, стоп на epoch 6 (4 без улучшения).
    assert summary["epochs_run"] == 7
    assert summary["val_loss_history"] == pytest.approx(script[:7])


def test_no_early_stop_when_improving(tmp_path):
    """STUB-COVERED: монотонно убывающий val-loss => без early-stop, все эпохи."""
    script = [0.9, 0.8, 0.7, 0.6, 0.5]
    summary, _bb, _out = _run(tmp_path, val_script=script, epochs=5, patience=2)
    assert summary["early_stopped"] is False
    assert summary["epochs_run"] == 5
    assert summary["best_epoch"] == 4


# ---------------------------------------------------------------------------
# 2. grad-accum: optimizer.step раз в grad_accum микро-батчей
# ---------------------------------------------------------------------------


def test_grad_accum_optimizer_step_cadence(tmp_path):
    """STUB-COVERED: ``optimizer.step`` ровно раз в ``grad_accum`` микро-батчей.

    4 train-батча, grad_accum=2 => 2 шага/эпоху. Один эпоха-прогон (val растёт
    сразу, patience=1 => стоп после epoch 0). Наблюдается через _FakeOptimizer.
    """
    summary, _bb, _out = _run(
        tmp_path,
        val_script=[0.5, 0.6, 0.7],
        epochs=3,
        grad_accum=2,
        train_batches=4,
        patience=1,
    )
    # epoch 0 = best (0.5); epoch 1 хуже => epochs_since_best=1>=patience => стоп.
    # значит ровно 2 эпохи прогнаны: 0 и 1.
    assert summary["epochs_run"] == 2
    # 4 батча / accum 2 = 2 шага за эпоху * 2 эпохи = 4 шага.
    assert summary["optimizer_steps"] == 4


def test_grad_accum_tail_step_on_partial(tmp_path):
    """STUB-COVERED: неполный хвост микро-батчей всё равно даёт optimizer.step.

    5 батчей, accum=2 => 2 полных шага + 1 хвостовой = 3 шага за эпоху.
    """
    summary, _bb, _out = _run(
        tmp_path,
        val_script=[0.5, 0.6],
        epochs=2,
        grad_accum=2,
        train_batches=5,
        patience=1,
    )
    assert summary["epochs_run"] == 2  # epoch0 best, epoch1 хуже => стоп
    assert summary["optimizer_steps"] == 6  # 3 шага * 2 эпохи


# ---------------------------------------------------------------------------
# 3. freeze -> unfreeze ровно на epoch == freeze_epochs
# ---------------------------------------------------------------------------


def test_freeze_then_unfreeze_at_exact_epoch(tmp_path):
    """STUB-COVERED: бэкбон заморожен для epoch<freeze_epochs, разморожен на ==.

    Наблюдаем флаг стаба ``frozen`` и счётчики freeze/unfreeze. freeze_epochs=3:
    эпохи 0,1,2 — frozen; на epoch 3 — unfreeze. Скрипт монотонно убывает, чтобы
    дойти до epoch 3+ без early-stop.
    """
    script = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    bb = _StubBackbone(feature_dim=8, seed=1)
    bb.freeze()  # старт замороженным (как build_backbone(freeze=True))
    summary, bb, _out = _run(
        tmp_path,
        val_script=script,
        epochs=6,
        freeze_epochs=3,
        patience=6,
        backbone=bb,
    )
    # к концу (epoch>=3) бэкбон разморожен ровно один раз.
    assert summary["epochs_run"] == 6
    assert bb.frozen is False
    assert bb.unfreeze_calls == 1  # ровно одно переключение на epoch==3


def test_backbone_stays_frozen_when_freeze_epochs_exceeds_run(tmp_path):
    """STUB-COVERED: freeze_epochs больше числа эпох => unfreeze не происходит."""
    script = [0.9, 0.8]
    bb = _StubBackbone(feature_dim=8, seed=1)
    bb.freeze()
    summary, bb, _out = _run(
        tmp_path,
        val_script=script,
        epochs=2,
        freeze_epochs=5,
        patience=5,
        backbone=bb,
    )
    assert summary["epochs_run"] == 2
    assert bb.frozen is True
    assert bb.unfreeze_calls == 0


# ---------------------------------------------------------------------------
# 4. best-epoch == min val-loss; чекпойнт соответствует
# ---------------------------------------------------------------------------


def test_best_epoch_is_min_and_checkpoint_matches(tmp_path):
    """STUB-COVERED: чекпойнт записан для эпохи с минимальным val-loss.

    Минимум на epoch 2; читаем meta.json и сверяем epoch + best_val_loss.
    """
    script = [0.9, 0.7, 0.4, 0.5, 0.6, 0.7]
    summary, _bb, out = _run(tmp_path, val_script=script, epochs=6, patience=3)
    assert summary["best_epoch"] == 2
    ckpt = load_checkpoint(out)
    assert ckpt["epoch"] == 2
    assert ckpt["extra"]["best_val_loss"] == pytest.approx(0.4)
    assert ckpt["extra"]["best_epoch"] == 2
    # provenance честный (синтетика без реального прогона).
    assert ckpt["provenance"] == "SYNTHETIC — not a species metric"
    assert ckpt["state_dict"] is None  # стаб не пишет веса


# ---------------------------------------------------------------------------
# 5. class_weight ветка: веса из распределения меток
# ---------------------------------------------------------------------------


def test_class_weight_branch_computes_weights_from_labels(tmp_path):
    """STUB-COVERED: при class_weight=True веса считаются из меток train-лоадера.

    Дисбаланс 3:1 (класс 0 чаще) => вес класса 0 < вес класса 1.
    """
    # 4 батча по 2 => 8 меток. 6 нулей, 2 единицы.
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    summary, _bb, _out = _run(
        tmp_path,
        val_script=[0.5, 0.6],
        epochs=2,
        patience=1,
        class_weight=True,
        train_batches=4,
        train_labels=labels,
    )
    weights = summary["class_weights"]
    assert weights is not None
    assert len(weights) == 2
    # редкий класс (1) весит больше частого (0).
    assert weights[1] > weights[0]


def test_class_weight_disabled_yields_none(tmp_path):
    """STUB-COVERED: class_weight=False => class_weights отсутствуют (None)."""
    summary, _bb, _out = _run(
        tmp_path, val_script=[0.5, 0.6], epochs=2, patience=1, class_weight=False
    )
    assert summary["class_weights"] is None


def test_compute_class_weights_balanced_and_zero_safe():
    """STUB-COVERED (unit): balanced-веса, класс без примеров => вес 0."""
    w = _compute_class_weights([0, 0, 0, 1], n_classes=3)
    assert w.shape == (3,)
    assert w[2] == 0.0  # класс 2 отсутствует
    assert w[1] > w[0]  # класс 1 реже класса 0


# ---------------------------------------------------------------------------
# 6. checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_restores_metadata(tmp_path):
    """STUB-COVERED: load(save(...)) восстанавливает vocab/feature_dim/model_name/provenance."""
    out = tmp_path / "ck"
    vocab = {"Parus_major": 0, "Turdus_merula": 1}
    save_checkpoint(
        out,
        state_dict=None,
        vocab=vocab,
        model_name="passt",
        feature_dim=768,
        epoch=7,
        provenance="SYNTHETIC — not a species metric",
        sr=32_000,
        clip_sec=10.0,
        extra={"best_val_loss": 0.123},
    )
    loaded = load_checkpoint(out)
    assert loaded["vocab"] == vocab
    assert loaded["model_name"] == "passt"
    assert loaded["feature_dim"] == 768
    assert loaded["epoch"] == 7
    assert loaded["provenance"] == "SYNTHETIC — not a species metric"
    assert loaded["sr"] == 32_000
    assert loaded["clip_sec"] == pytest.approx(10.0)
    assert loaded["extra"]["best_val_loss"] == pytest.approx(0.123)
    assert loaded["state_dict"] is None


# ---------------------------------------------------------------------------
# 7. resume continues from saved epoch
# ---------------------------------------------------------------------------


def test_resume_continues_from_saved_epoch(tmp_path):
    """STUB-COVERED: finetune(resume=ckpt) стартует с saved_epoch+1.

    Сохраняем чекпойнт на epoch 4; resume => start_epoch == 5, история val-loss
    индексируется с epoch 5.
    """
    resume_ck = tmp_path / "resume_ck"
    save_checkpoint(
        resume_ck,
        state_dict=None,
        vocab={"a": 0, "b": 1},
        model_name="passt",
        feature_dim=8,
        epoch=4,
        provenance="SYNTHETIC — not a species metric",
        extra={"best_val_loss": 0.5, "best_epoch": 4},
    )
    # скрипт длиной 8: индексы 5,6,7 будут читаться.
    script = [9, 9, 9, 9, 9, 0.4, 0.3, 0.2]
    summary, _bb, out = _run(
        tmp_path,
        val_script=script,
        epochs=8,
        patience=8,
        resume=resume_ck,
    )
    assert summary["start_epoch"] == 5
    # прогнаны эпохи 5,6,7 => 3 прогона.
    assert summary["epochs_run"] == 3
    assert summary["val_loss_history"] == pytest.approx([0.4, 0.3, 0.2])
    assert summary["best_epoch"] == 7


# ---------------------------------------------------------------------------
# 8. ЕДИНСТВЕННЫЙ torch-тест: реальный fwd/bwd на tiny SpeciesHead
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_real_forward_backward_updates_param(tmp_path):
    """TORCH-ONLY: реальный forward + loss.backward() + optimizer.step меняет параметр.

    Локально torch ЕСТЬ => тест РАБОТАЕТ; в CI без torch — SKIP (единственный
    допустимый рост skipped). Стаб этого не покрывает (нет тензорных операций).
    """
    torch = pytest.importorskip("torch")
    from faun.training.backbones import SpeciesHead

    feature_dim, n_classes = 8, 3
    head = SpeciesHead(feature_dim, n_classes)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)

    features = torch.randn(4, feature_dim)
    labels = torch.tensor([0, 1, 2, 0])

    before = head.linear.weight.detach().clone()
    logits = head(features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    after = head.linear.weight.detach()

    assert not torch.allclose(before, after), "параметр головы должен измениться"
    assert logits.shape == (4, n_classes)
