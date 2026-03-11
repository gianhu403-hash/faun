"""Tests for YDB microphone repository — scan_query usage."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cloud.db.microphones import Microphone
from cloud.db.ydb_microphones import YDBMicrophoneRepository


def _make_row(**kwargs):
    defaults = dict(
        id=1,
        mic_uid="MIC-0001",
        lat=57.0,
        lon=45.0,
        zone_type="core",
        sub_district="north",
        status="online",
        battery_pct=95.0,
        district_slug="varnavino",
        installed_at="2026-01-15",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_scan_result(rows):
    """Simulate a single scan_query result part."""
    return SimpleNamespace(result_set=SimpleNamespace(rows=rows))


class TestGetOnlineUseScanQuery:
    """get_online() must use scan_query (not transaction.execute) to avoid
    TruncatedResponseError on >1000 rows."""

    @patch("cloud.db.ydb_client.get_driver")
    def test_scan_query_called(self, mock_get_driver):
        """scan_query is used instead of session pool / execute_query."""
        rows = [_make_row(id=i, mic_uid=f"MIC-{i:04d}") for i in range(3)]
        mock_driver = MagicMock()
        mock_driver.table_client.scan_query.return_value = iter(
            [_make_scan_result(rows)]
        )
        mock_get_driver.return_value = mock_driver

        repo = YDBMicrophoneRepository()
        result = repo.get_online(limit=2000)

        mock_driver.table_client.scan_query.assert_called_once()
        query_arg = mock_driver.table_client.scan_query.call_args[0][0]
        assert "status = 'online'" in query_arg
        assert "LIMIT 2000" in query_arg
        assert len(result) == 3
        assert all(isinstance(m, Microphone) for m in result)

    @patch("cloud.db.ydb_client.get_driver")
    def test_multiple_result_parts(self, mock_get_driver):
        """scan_query may return results in multiple parts — all are collected."""
        part1 = _make_scan_result([_make_row(id=1)])
        part2 = _make_scan_result([_make_row(id=2)])
        mock_driver = MagicMock()
        mock_driver.table_client.scan_query.return_value = iter([part1, part2])
        mock_get_driver.return_value = mock_driver

        repo = YDBMicrophoneRepository()
        result = repo.get_online()

        assert len(result) == 2

    @patch("cloud.db.ydb_client.get_driver")
    def test_empty_result(self, mock_get_driver):
        """No online mics → empty list."""
        mock_driver = MagicMock()
        mock_driver.table_client.scan_query.return_value = iter([])
        mock_get_driver.return_value = mock_driver

        repo = YDBMicrophoneRepository()
        result = repo.get_online()

        assert result == []
