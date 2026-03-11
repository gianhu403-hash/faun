"""Tests for _classify_via_edge() — edge HTTP API client with retry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from edge.audio.classifier import AudioResult


def _make_mock_response(label: str = "chainsaw", confidence: float = 0.87) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = {
        "label": label,
        "confidence": confidence,
        "raw_scores": {label: confidence},
    }
    return resp


@pytest.fixture()
def wav_file(tmp_path: Path) -> str:
    """Create a dummy WAV file for testing."""
    p = tmp_path / "test.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    return str(p)


class TestClassifyViaEdgeSuccess:
    def test_classify_via_edge_success(self, wav_file: str) -> None:
        """Successful edge response returns correct AudioResult."""
        mock_resp = _make_mock_response("chainsaw", 0.87)

        with patch("httpx.post", return_value=mock_resp):
            from cloud.interface.main import _classify_via_edge

            result = _classify_via_edge(wav_file)

        assert isinstance(result, AudioResult)
        assert result.label == "chainsaw"
        assert result.confidence == 0.87


class TestClassifyViaEdgeRetry:
    def test_classify_via_edge_retry_on_connect_error(self, wav_file: str) -> None:
        """On ConnectError, retry once and return result if second attempt succeeds."""
        mock_resp = _make_mock_response("chainsaw", 0.87)

        with patch(
            "httpx.post", side_effect=[httpx.ConnectError("refused"), mock_resp]
        ):
            from cloud.interface.main import _classify_via_edge

            result = _classify_via_edge(wav_file)

        assert result.label == "chainsaw"
        assert result.confidence == 0.87

    def test_classify_via_edge_double_failure_returns_unknown(
        self, wav_file: str
    ) -> None:
        """Two consecutive ConnectErrors return unknown without crashing."""
        with patch(
            "httpx.post",
            side_effect=[httpx.ConnectError("refused"), httpx.ConnectError("refused")],
        ):
            from cloud.interface.main import _classify_via_edge

            result = _classify_via_edge(wav_file)

        assert result.label == "unknown"
        assert result.confidence == 0.0
