"""Shared fixtures for the active pipeline test suite (tests/).

After the v2 reorg this suite covers only the REUSE ML core that moved into
``faun/ml`` (onset detector + YAMNet classifier). TDOA / triangulation and
all hackathon fixtures live in legacy/tests/conftest.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from faun.ml.yamnet import AudioResult, AudioClass  # noqa: F401  (re-export for tests)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate used across the project."""
    return 16000


# ---------------------------------------------------------------------------
# YAMNet mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_yamnet_model() -> MagicMock:
    """MagicMock that behaves like TF-Hub YAMNet.

    Returns:
        A callable mock producing (scores, embeddings, spectrogram) where
        embeddings.numpy() yields shape (5, 1024).
    """
    yamnet = MagicMock(name="yamnet")

    scores = MagicMock(name="scores")
    scores.numpy.return_value = np.random.rand(5, 521).astype(np.float32)

    embeddings = MagicMock(name="embeddings")
    embeddings.numpy.return_value = np.random.randn(5, 1024).astype(np.float32)

    spectrogram = MagicMock(name="spectrogram")
    spectrogram.numpy.return_value = np.random.rand(5, 64).astype(np.float32)

    yamnet.return_value = (scores, embeddings, spectrogram)
    return yamnet


@pytest.fixture
def mock_head_model() -> MagicMock:
    """MagicMock that behaves like the Keras classification head.

    input_shape is (None, 2181) and __call__ returns a 6-class softmax-like
    prediction with class 0 (chainsaw) having the highest score by default.
    """
    head = MagicMock(name="head")
    head.input_shape = (None, 2181)

    default_pred = np.array([[0.60, 0.10, 0.08, 0.07, 0.05, 0.10]], dtype=np.float32)
    pred_tensor = MagicMock(name="pred_tensor")
    pred_tensor.numpy.return_value = default_pred
    head.return_value = pred_tensor

    return head
