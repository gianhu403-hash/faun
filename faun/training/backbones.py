"""Бэкбоны fine-tune + классификационная голова + numpy-стаб для тестов.

Три уровня:

* :class:`Backbone` — структурный протокол (``feature_dim`` + ``forward``);
* :func:`build_backbone` — фабрика реального бэкбона (PaSST по умолчанию; AST /
  BEATs подключаемы), torch+веса тянутся **лениво**;
* :class:`SpeciesHead` — линейная голова ``feature_dim -> n_classes`` (torch
  ``nn.Module``, создаётся лениво);
* :class:`_StubBackbone` — **чистый numpy**, детерминированный (seeded), без
  torch. Имеет ``feature_dim``, ``forward`` и наблюдаемый флаг ``frozen``
  (а также счётчик ``forward_calls``) — на нём гоняется весь control-flow лупа
  TF/torch-free.

ЛИЦЕНЗИИ бэкбонов (см. ``docs/finetuning.md``):
* PaSST (``hear21passt``) — код Apache-2.0; веса обучены на AudioSet (оговорка о
  датасете) — продуктовый дефолт;
* AST (HF ``MIT/ast-finetuned-audioset-10-10-0.4593``) — код BSD-3;
* BEATs — код MIT, но веса под кастомной MS-лицензией (gotcha — проверять перед
  коммерцией).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    pass

# feature-dim бэкбонов в режиме "embed_only".
# ВНИМАНИЕ: PaSST scene_embedding = 1295 (527 logits + 768 features) — для головы
# берём ТОЛЬКО 768, не 1295.
_PASST_FEATURE_DIM = 768
_AST_FEATURE_DIM = 768
_BEATS_FEATURE_DIM = 768

_KNOWN_BACKBONES = {
    "passt": _PASST_FEATURE_DIM,
    "ast": _AST_FEATURE_DIM,
    "beats": _BEATS_FEATURE_DIM,
    "stub": 16,
}

# Ревизия весов AST на HF Hub: оператор закрепляет конкретный коммит через env
# на кластере (воспроизводимость + B615); дефолт "main" — для смоук-выкатки.
import os as _os  # noqa: E402 - локальный импорт констант, stdlib

_AST_REVISION = _os.environ.get("FAUN_AST_REVISION", "main")


@runtime_checkable
class Backbone(Protocol):
    """Контракт бэкбона: raw-batch -> фиксированные фичи [B, feature_dim]."""

    feature_dim: int

    def forward(self, batch: Any) -> Any:
        """Вернуть фичи ``[B, feature_dim]`` для батча сырых сигналов."""
        ...


class _StubBackbone:
    """Чистый numpy-стаб бэкбона: детерминированный, наблюдаемый, без torch.

    Назначение — гонять control-flow лупа (epoch loop, freeze/unfreeze,
    grad-accum, early-stop, best-epoch, checkpoint, resume) полностью TF/torch-free.

    Наблюдаемые атрибуты:
    * ``frozen`` — флаг заморозки; луп должен ставить ``True`` пока
      ``epoch < freeze_epochs`` и переключать на ``False`` ровно на
      ``epoch == freeze_epochs`` (через :meth:`freeze`/:meth:`unfreeze`);
    * ``freeze_calls`` / ``unfreeze_calls`` — счётчики переключений;
    * ``forward_calls`` — сколько раз вызван ``forward``.

    ``forward`` детерминирован: фичи — функция от seed и хэша содержимого батча.
    """

    def __init__(self, *, feature_dim: int = 16, seed: int = 0) -> None:
        self.feature_dim = int(feature_dim)
        self._seed = int(seed)
        self.frozen = False
        self.freeze_calls = 0
        self.unfreeze_calls = 0
        self.forward_calls = 0

    def freeze(self) -> None:
        """Заморозить бэкбон (наблюдаемо лупом)."""
        self.frozen = True
        self.freeze_calls += 1

    def unfreeze(self) -> None:
        """Разморозить бэкбон (наблюдаемо лупом)."""
        self.frozen = False
        self.unfreeze_calls += 1

    def forward(self, batch: Any) -> np.ndarray:
        """Детерминированные фичи [B, feature_dim] из numpy-батча.

        ``batch`` — массив-подобное ``[B, win]`` (или список сигналов). Фичи
        строятся seeded-ГСЧ от суммы сигнала, поэтому одинаковый вход даёт
        одинаковый выход — control-flow воспроизводим.
        """
        self.forward_calls += 1
        arr = np.asarray(batch, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        b = arr.shape[0]
        feats = np.empty((b, self.feature_dim), dtype=np.float32)
        for i in range(b):
            key = self._seed + int(abs(np.sum(arr[i])) * 1e3) % 2**31
            rng = np.random.default_rng(key)
            feats[i] = rng.standard_normal(self.feature_dim).astype(np.float32)
        return feats


def build_backbone(
    name: str = "passt",
    *,
    sr: int = 32_000,
    win_s: float = 10.0,
    freeze: bool = True,
) -> Backbone:
    """Построить реальный бэкбон (torch+веса лениво). FROZEN сигнатура.

    ``name``:
    * ``"passt"`` — ``hear21passt`` (нативно 32 кГц, свой mel-фронтенд из raw),
      режим ``embed_only`` => feature_dim=768;
    * ``"ast"`` — HF AST (BSD-3);
    * ``"beats"`` — BEATs (внимание на лицензию весов);
    * ``"stub"`` — :class:`_StubBackbone` (numpy, без torch; для тестов/смоука).

    При ``freeze=True`` параметры бэкбона замораживаются (``requires_grad=False``)
    — луп размораживает их позже (``freeze_epochs``).
    """
    if name == "stub":
        stub = _StubBackbone(feature_dim=_KNOWN_BACKBONES["stub"])
        if freeze:
            stub.freeze()
        return stub

    if name not in _KNOWN_BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}; known: {sorted(_KNOWN_BACKBONES)}"
        )

    if name == "passt":
        return _build_passt(sr=sr, win_s=win_s, freeze=freeze)
    if name == "ast":
        return _build_ast(freeze=freeze)
    if name == "beats":  # pragma: no cover - требует кастомных весов BEATs
        raise NotImplementedError(
            "BEATs weights ship under a custom Microsoft license; wire it on "
            "cluster after the license check (see docs/finetuning.md)."
        )
    raise ValueError(f"unhandled backbone {name!r}")  # pragma: no cover


def _build_passt(*, sr: int, win_s: float, freeze: bool) -> Backbone:
    """PaSST через ``hear21passt`` (torch+timm лениво; только на кластере)."""
    import torch  # noqa: F401 - нужен для requires_grad ниже
    from hear21passt.base import get_basic_model

    # PaSST нативно 32 кГц и сам гонит mel из raw-waveform [B, sec*32000].
    if sr != 32_000:  # pragma: no cover - конфиг-гард
        raise ValueError(f"PaSST is native 32kHz; got sr={sr} (upsample upstream)")

    model = get_basic_model(mode="embed_only")

    class _PasstBackbone:
        feature_dim = _PASST_FEATURE_DIM

        def __init__(self, net: Any) -> None:
            self.net = net
            # Наблюдаемый флаг (контракт стаба): луп читает frozen на реальном пути.
            self.frozen = False

        def freeze(self) -> None:
            for p in self.net.parameters():
                p.requires_grad = False
            self.frozen = True

        def unfreeze(self) -> None:
            for p in self.net.parameters():
                p.requires_grad = True
            self.frozen = False

        def forward(self, batch: Any) -> Any:
            return self.net(batch)

    bb = _PasstBackbone(model)
    if freeze:
        bb.freeze()
    return bb


def _build_ast(*, freeze: bool) -> Backbone:  # pragma: no cover - требует HF+torch
    """AST через HuggingFace ``transformers`` (BSD-3; torch лениво)."""
    import torch  # noqa: F401
    from transformers import ASTModel

    # nosec B615: revision закрепляется оператором на кластере под нужный
    # коммит при ручной выкатке AST-бэкбона (опциональный путь, в CI не
    # исполняется). Конкретный SHA не хардкодим, чтобы не пинить мёртвую ревизию.
    model = ASTModel.from_pretrained(  # nosec B615
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        revision=_AST_REVISION,
    )

    class _AstBackbone:
        feature_dim = _AST_FEATURE_DIM

        def __init__(self, net: Any) -> None:
            self.net = net
            # Наблюдаемый флаг (контракт стаба): луп читает frozen на реальном пути.
            self.frozen = False

        def freeze(self) -> None:
            for p in self.net.parameters():
                p.requires_grad = False
            self.frozen = True

        def unfreeze(self) -> None:
            for p in self.net.parameters():
                p.requires_grad = True
            self.frozen = False

        def forward(self, batch: Any) -> Any:
            return self.net(batch).pooler_output

    bb = _AstBackbone(model)
    if freeze:
        bb.freeze()
    return bb


def _species_head_base():
    """Лениво вернуть ``torch.nn.Module`` либо ``object`` (torch отсутствует)."""
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        return object
    return nn.Module


class SpeciesHead:
    """Линейная голова ``feature_dim -> n_classes`` (torch ``nn.Module``).

    torch тянется **лениво** при инстанцировании; класс динамически наследует
    ``nn.Module`` тогда же, поэтому импорт модуля torch-free. На машине без torch
    инстанцирование явно падает ``RuntimeError`` (control-flow тесты сюда не
    заходят — голову заменяет numpy-стаб/инжекция).
    """

    def __init__(self, feature_dim: int, n_classes: int) -> None:
        try:
            import torch.nn as nn
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SpeciesHead requires PyTorch (requirements-train.txt). "
                "Control-flow tests use the numpy stub / injection hooks."
            ) from exc

        base = _species_head_base()
        if base is not object and type(self) is SpeciesHead:
            self.__class__ = _head_subclass(base)

        base.__init__(self)  # type: ignore[misc]
        self.feature_dim = int(feature_dim)
        self.n_classes = int(n_classes)
        self.linear = nn.Linear(feature_dim, n_classes)

    def forward(self, features: Any) -> Any:
        """Логиты ``[B, n_classes]`` из фич ``[B, feature_dim]``."""
        return self.linear(features)


_HEAD_SUBCLASS: dict[type, type] = {}


def _head_subclass(nn_module: type) -> type:
    """Создать (один раз) подкласс ``SpeciesHead``, наследующий ``nn.Module``."""
    cached = _HEAD_SUBCLASS.get(nn_module)
    if cached is not None:
        return cached
    new_cls = type("SpeciesHead", (SpeciesHead, nn_module), {})
    _HEAD_SUBCLASS[nn_module] = new_cls
    return new_cls
