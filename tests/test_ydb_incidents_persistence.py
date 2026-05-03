"""Tests for FAUN-38a B2 — YDB persistence of drone photo + 3 other fields.

These tests verify that ``cloud.db.ydb_incidents`` correctly persists fields
that today are silently dropped by ``update_incident`` because they are not
in ``_YDB_PERSISTABLE``:

- ``drone_photo_b64`` (raw PNG bytes from drone)
- ``drone_comment`` (Gemma 3 vision analysis)
- ``ranger_photo_b64`` (photo uploaded by ranger on site)
- ``alert_message_ids`` (mapping chat_id -> Telegram message_id, for edits)

Failure mode today: ``update_incident(id, drone_photo_b64=b"...")`` returns
silently because the field is filtered out at line 186 (``ydb_fields = {k: v
for k, v in fields.items() if k in _YDB_PERSISTABLE}``) and the function
hits the early return at line 188 (``if not ydb_fields: return``). The drone
photo is therefore lost on restart.

These tests MUST fail until ``_YDB_PERSISTABLE``/``_FIELD_TYPES`` are extended
and a warning is emitted for unknown fields.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from cloud.db.ydb_incidents import (
    _FIELD_TYPES,
    _YDB_PERSISTABLE,
    YDBIncidentRepository,
)


# ---------------------------------------------------------------------------
# (a) _YDB_PERSISTABLE membership
# ---------------------------------------------------------------------------


class TestYdbPersistableMembership:
    """The 4 fields silently dropped today must be declared persistable."""

    def test_ydb_persistable_includes_drone_photo_b64(self):
        assert "drone_photo_b64" in _YDB_PERSISTABLE

    def test_ydb_persistable_includes_drone_comment(self):
        assert "drone_comment" in _YDB_PERSISTABLE

    def test_ydb_persistable_includes_ranger_photo_b64(self):
        assert "ranger_photo_b64" in _YDB_PERSISTABLE

    def test_ydb_persistable_includes_alert_message_ids(self):
        assert "alert_message_ids" in _YDB_PERSISTABLE


# ---------------------------------------------------------------------------
# (d) _FIELD_TYPES mappings — YDB column types must be correct
# ---------------------------------------------------------------------------


class TestFieldTypesDeclarations:
    """Each new persistable field must have a YDB type declared."""

    def test_field_types_declares_drone_photo_b64_as_string(self):
        # raw PNG bytes -> YDB ``String`` (binary blob), not ``Utf8``
        assert _FIELD_TYPES["drone_photo_b64"] == "String"

    def test_field_types_declares_alert_message_ids_as_utf8(self):
        # dict[int, int] is serialised to JSON before persistence -> ``Utf8``
        assert _FIELD_TYPES["alert_message_ids"] == "Utf8"


# ---------------------------------------------------------------------------
# (b) update_incident actually includes new fields in SQL params
# ---------------------------------------------------------------------------


class TestUpdateIncidentPersistsNewFields:
    """``update_incident`` must reach ``execute_query`` with the new params."""

    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.__init__", return_value=None)
    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.get_incident")
    @patch("cloud.db.ydb_client.get_pool")
    def test_update_incident_persists_drone_photo_b64(
        self,
        mock_get_pool,
        mock_get_incident,
        mock_init,
    ):
        # Mock current state so state machine validation passes
        # (no status change here, but get_incident may still be called).
        mock_get_incident.return_value = MagicMock(status="pending")

        captured: dict = {}

        def fake_execute_query(session, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params or {}
            return [MagicMock(rows=[])]

        # pool.retry_operation_sync(_upd) calls _upd(session); _upd calls
        # execute_query. Forward to our fake by invoking the closure with a
        # mock session.
        def run_op(fn):
            return fn(MagicMock())

        mock_pool = MagicMock()
        mock_pool.retry_operation_sync.side_effect = run_op
        mock_get_pool.return_value = mock_pool

        with patch("cloud.db.ydb_client.execute_query", side_effect=fake_execute_query):
            repo = YDBIncidentRepository()
            repo.update_incident("id1", drone_photo_b64=b"PNGDATA")

        assert "params" in captured, (
            "execute_query was never called -- update_incident silently "
            "dropped drone_photo_b64 because it is not in _YDB_PERSISTABLE"
        )
        assert "$drone_photo_b64" in captured["params"]
        assert captured["params"]["$drone_photo_b64"] == b"PNGDATA"
        assert "drone_photo_b64 = $drone_photo_b64" in captured["sql"]

    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.__init__", return_value=None)
    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.get_incident")
    @patch("cloud.db.ydb_client.get_pool")
    def test_update_incident_persists_alert_message_ids_as_json(
        self,
        mock_get_pool,
        mock_get_incident,
        mock_init,
    ):
        """``alert_message_ids: dict[int, int]`` must be JSON-serialised
        before it reaches YDB (``Utf8`` column cannot hold a dict)."""
        import json

        mock_get_incident.return_value = MagicMock(status="pending")

        captured: dict = {}

        def fake_execute_query(session, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params or {}
            return [MagicMock(rows=[])]

        def run_op(fn):
            return fn(MagicMock())

        mock_pool = MagicMock()
        mock_pool.retry_operation_sync.side_effect = run_op
        mock_get_pool.return_value = mock_pool

        with patch("cloud.db.ydb_client.execute_query", side_effect=fake_execute_query):
            repo = YDBIncidentRepository()
            repo.update_incident("id1", alert_message_ids={123: 456, 789: 100})

        assert "params" in captured, (
            "execute_query was never called -- alert_message_ids was filtered "
            "out by _YDB_PERSISTABLE"
        )
        assert "$alert_message_ids" in captured["params"]
        serialised = captured["params"]["$alert_message_ids"]
        assert isinstance(serialised, str), (
            "alert_message_ids must be serialised to a JSON string before "
            "being passed as a Utf8 param"
        )
        # Round-trip the JSON to compare semantically (key ordering free).
        assert json.loads(serialised) == {"123": 456, "789": 100}


# ---------------------------------------------------------------------------
# (c) update_incident logs a warning when unknown fields are dropped
# ---------------------------------------------------------------------------


class TestUpdateIncidentWarnsOnUnknownFields:
    """Today's silent drop hides bugs (the entire FAUN-38a issue). Any field
    not in ``_YDB_PERSISTABLE`` must produce a WARNING log so callers can see
    it during development."""

    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.__init__", return_value=None)
    @patch("cloud.db.ydb_incidents.YDBIncidentRepository.get_incident")
    @patch("cloud.db.ydb_client.get_pool")
    def test_update_incident_logs_warning_for_unknown_fields(
        self,
        mock_get_pool,
        mock_get_incident,
        mock_init,
        caplog: pytest.LogCaptureFixture,
    ):
        mock_get_incident.return_value = MagicMock(status="pending")

        def fake_execute_query(session, sql, params=None):
            return [MagicMock(rows=[])]

        def run_op(fn):
            return fn(MagicMock())

        mock_pool = MagicMock()
        mock_pool.retry_operation_sync.side_effect = run_op
        mock_get_pool.return_value = mock_pool

        with caplog.at_level(logging.WARNING, logger="cloud.db.ydb_incidents"):
            with patch(
                "cloud.db.ydb_client.execute_query",
                side_effect=fake_execute_query,
            ):
                repo = YDBIncidentRepository()
                repo.update_incident("id1", status="accepted", bogus_field="x")

        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert warnings, (
            "no WARNING was logged when update_incident dropped "
            "'bogus_field' -- silent drops hide bugs (this is exactly how "
            "drone_photo_b64 was lost)"
        )
        joined = " ".join(rec.getMessage() for rec in warnings).lower()
        assert (
            "bogus_field" in joined
            or "non-persistable" in joined
            or ("dropping" in joined)
        ), (
            "warning message should name the unknown field or use the word "
            f"'non-persistable'/'dropping'; got: {joined!r}"
        )


# ---------------------------------------------------------------------------
# (d) write/read symmetry — _row_to_incident must surface persisted fields
# (regression test for FAUN-38a F2 silent-failure-hunter finding)
# ---------------------------------------------------------------------------


class TestRowToIncidentSurfacesNewFields:
    """If _YDB_PERSISTABLE writes a field, _row_to_incident MUST read it back.

    Without this regression: write succeeds, restart, read silently returns
    Incident with empty drone_photo_b64/comment/etc -- claim of persistence
    is false.
    """

    def test_row_to_incident_reads_drone_photo_b64(self):
        from types import SimpleNamespace

        row = SimpleNamespace(
            id="x",
            audio_class="chainsaw",
            lat=57.3,
            lon=44.6,
            confidence=0.9,
            gating_level="alert",
            status="pending",
            created_at=1234567890.0,
            district="",
            accepted_by_chat_id=0,
            accepted_by_name="",
            accepted_at=None,
            arrived_at=None,
            response_time_min=None,
            ranger_report_raw=None,
            ranger_report_legal=None,
            resolution_details="",
            is_demo=False,
            drone_photo_b64=b"PNGDATA",
            drone_comment="дрон видел рубку",
            ranger_photo_b64=b"JPEGDATA",
            alert_message_ids='{"123": 456, "789": 100}',
        )
        incident = YDBIncidentRepository._row_to_incident(row)
        assert incident.drone_photo_b64 == b"PNGDATA"
        assert incident.drone_comment == "дрон видел рубку"
        assert incident.ranger_photo_b64 == b"JPEGDATA"
        assert incident.alert_message_ids == {123: 456, 789: 100}

    def test_row_to_incident_handles_missing_alert_message_ids(self):
        """Empty/missing alert_message_ids should give empty dict, not crash."""
        from types import SimpleNamespace

        row = SimpleNamespace(
            id="x",
            audio_class="chainsaw",
            lat=57.3,
            lon=44.6,
            confidence=0.9,
            gating_level="alert",
            status="pending",
            created_at=0.0,
            district="",
            accepted_by_chat_id=0,
            accepted_by_name="",
            accepted_at=None,
            arrived_at=None,
            response_time_min=None,
            ranger_report_raw=None,
            ranger_report_legal=None,
            resolution_details="",
            is_demo=False,
            drone_photo_b64=None,
            drone_comment=None,
            ranger_photo_b64=None,
            alert_message_ids=None,
        )
        incident = YDBIncidentRepository._row_to_incident(row)
        assert incident.alert_message_ids == {}
        assert incident.drone_photo_b64 is None

    def test_update_incident_rejects_non_dict_alert_message_ids(self):
        """alert_message_ids=list silently coerced to '' would break read side.
        Must raise TypeError loudly instead.
        """
        with (
            patch.object(
                YDBIncidentRepository,
                "get_incident",
                return_value=MagicMock(status="pending"),
            ),
            patch("cloud.db.ydb_client.get_pool"),
        ):
            repo = YDBIncidentRepository()
            with pytest.raises(TypeError, match="alert_message_ids must be dict"):
                repo.update_incident("id1", alert_message_ids=[1, 2, 3])

    def test_update_incident_accepts_none_alert_message_ids_as_empty_map(self):
        """None should write '{}' (explicit empty), distinguishable from broken."""
        captured = {}

        def fake_execute_query(session, sql, params=None):
            captured["params"] = params
            return [MagicMock(rows=[])]

        def run_op(fn):
            return fn(MagicMock())

        mock_pool = MagicMock()
        mock_pool.retry_operation_sync.side_effect = run_op

        with (
            patch.object(
                YDBIncidentRepository,
                "get_incident",
                return_value=MagicMock(status="pending"),
            ),
            patch("cloud.db.ydb_client.get_pool", return_value=mock_pool),
            patch("cloud.db.ydb_client.execute_query", side_effect=fake_execute_query),
        ):
            repo = YDBIncidentRepository()
            repo.update_incident("id1", alert_message_ids=None)

        assert captured["params"]["$alert_message_ids"] == "{}"
