"""Tests for FAUN-38a B1 -- YDBIncidentRepository missing methods.

These tests verify that ``YDBIncidentRepository`` implements
``get_stale_incidents`` and ``get_recent_nearby_incident`` with **the same
signatures** as the in-memory functions in ``cloud.db.incidents``.

Currently the YDB repository does NOT implement these methods.  The fallback
in ``cloud/db/incidents.py:219-222`` silently routes calls back to the
in-memory versions via ``getattr(_repo, ...)``, which operate on an empty
``_incidents = {}`` dict in YDB mode -- always returning ``[]`` / ``None``
(silent failure).

The tests deliberately call methods **directly** on a
``YDBIncidentRepository`` instance to bypass the ``getattr`` fallback and
prove the YDB class itself is missing the implementations.

Mock pattern: patch ``cloud.db.ydb_client.get_pool`` with a MagicMock whose
``retry_operation_sync`` invokes the passed function with a mock session,
and patch ``cloud.db.ydb_client.execute_query`` to return the desired rows.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cloud.db.incidents import Incident
from cloud.db.ydb_incidents import YDBIncidentRepository


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    id: str = "x",
    audio_class: str = "chainsaw",
    lat: float = 57.3,
    lon: float = 44.6,
    confidence: float = 0.9,
    gating_level: str = "alert",
    status: str = "pending",
    created_at: float = 0.0,
    accepted_at: float | None = None,
    accepted_by_chat_id: int = 0,
    accepted_by_name: str = "",
    district: str = "",
    arrived_at: float = 0.0,
    response_time_min: float = 0.0,
    ranger_report_raw: str = "",
    ranger_report_legal: str = "",
    resolution_details: str = "",
    is_demo: bool = False,
):
    """Factory for a YDB result row mock matching the ``incidents`` schema."""
    return SimpleNamespace(
        id=id,
        audio_class=audio_class,
        lat=lat,
        lon=lon,
        confidence=confidence,
        gating_level=gating_level,
        status=status,
        created_at=created_at,
        accepted_at=accepted_at,
        accepted_by_chat_id=accepted_by_chat_id,
        accepted_by_name=accepted_by_name,
        district=district,
        arrived_at=arrived_at,
        response_time_min=response_time_min,
        ranger_report_raw=ranger_report_raw,
        ranger_report_legal=ranger_report_legal,
        resolution_details=resolution_details,
        is_demo=is_demo,
    )


def _result(rows):
    """Wrap rows into the ``[result_set.rows]`` shape that ``execute_query`` returns."""
    rs = MagicMock()
    rs.rows = rows
    return [rs]


def _make_pool_mock():
    """Build a pool mock whose ``retry_operation_sync(fn)`` calls ``fn(session)``.

    The ``session`` is itself a MagicMock so the inner ``_q(session)`` lambda
    can pass it to ``execute_query`` (which is patched separately).
    """
    pool = MagicMock()
    pool.retry_operation_sync.side_effect = lambda fn: fn(MagicMock())
    return pool


@pytest.fixture(autouse=True)
def _patch_ensure_tables():
    """Stop ``YDBIncidentRepository.__init__`` from contacting real YDB."""
    with patch("cloud.db.ydb_client.ensure_tables"):
        yield


# ---------------------------------------------------------------------------
# 1-2. Existence checks (bypass getattr fallback in cloud/db/incidents.py)
# ---------------------------------------------------------------------------


def test_ydb_repo_has_get_stale_incidents_method():
    """YDBIncidentRepository must define get_stale_incidents itself.

    The fallback in cloud/db/incidents.py uses ``getattr(_repo, ...)`` which
    masks the missing method by returning the in-memory function instead --
    so this test pokes the class directly.
    """
    assert hasattr(YDBIncidentRepository, "get_stale_incidents"), (
        "YDBIncidentRepository is missing get_stale_incidents -- "
        "the cloud/db/incidents.py:219 getattr fallback is silently routing "
        "calls to the in-memory empty _incidents dict in YDB mode."
    )


def test_ydb_repo_has_get_recent_nearby_incident_method():
    """YDBIncidentRepository must define get_recent_nearby_incident itself."""
    assert hasattr(YDBIncidentRepository, "get_recent_nearby_incident"), (
        "YDBIncidentRepository is missing get_recent_nearby_incident -- "
        "the cloud/db/incidents.py:220 getattr fallback is silently routing "
        "calls to the in-memory empty _incidents dict in YDB mode."
    )


# ---------------------------------------------------------------------------
# 3-4. get_stale_incidents behaviour
# ---------------------------------------------------------------------------


@patch("cloud.db.ydb_client.execute_query")
@patch("cloud.db.ydb_client.get_pool")
def test_get_stale_incidents_returns_pending_older_than_threshold(
    mock_get_pool, mock_execute_query
):
    """Returns only pending incidents older than pending_max_age and accepted
    incidents older than accepted_max_age.

    Mock dataset (4 rows):
      - pending, created 2000s ago      -> stale (age > 1800)
      - pending, created   100s ago     -> fresh
      - accepted, accepted 4000s ago    -> stale (age > 3600)
      - accepted, accepted  100s ago    -> fresh
    Expected result: 2 Incident instances (the two old ones).
    """
    now = time.time()
    rows = [
        _row(id="old-pending", status="pending", created_at=now - 2000),
        _row(id="fresh-pending", status="pending", created_at=now - 100),
        _row(
            id="old-accepted",
            status="accepted",
            created_at=now - 5000,
            accepted_at=now - 4000,
        ),
        _row(
            id="fresh-accepted",
            status="accepted",
            created_at=now - 200,
            accepted_at=now - 100,
        ),
    ]
    mock_get_pool.return_value = _make_pool_mock()
    mock_execute_query.return_value = _result(rows)

    repo = YDBIncidentRepository()
    result = repo.get_stale_incidents(pending_max_age=1800, accepted_max_age=3600)

    assert isinstance(result, list)
    assert all(isinstance(i, Incident) for i in result)

    returned_ids = {i.id for i in result}
    assert returned_ids == {"old-pending", "old-accepted"}, (
        f"expected stale ids {{'old-pending', 'old-accepted'}}, got {returned_ids}"
    )


@patch("cloud.db.ydb_client.execute_query")
@patch("cloud.db.ydb_client.get_pool")
def test_get_stale_incidents_default_thresholds(mock_get_pool, mock_execute_query):
    """Defaults must match in-memory: pending_max_age=1800, accepted_max_age=3600.

    Asserts behaviour, not signature introspection: a pending row 1700s old
    must NOT be stale, while a pending row 1900s old must be stale.
    """
    now = time.time()
    rows = [
        _row(id="just-under", status="pending", created_at=now - 1700),
        _row(id="just-over", status="pending", created_at=now - 1900),
        _row(
            id="accepted-just-under",
            status="accepted",
            created_at=now - 4000,
            accepted_at=now - 3500,
        ),
        _row(
            id="accepted-just-over",
            status="accepted",
            created_at=now - 4000,
            accepted_at=now - 3700,
        ),
    ]
    mock_get_pool.return_value = _make_pool_mock()
    mock_execute_query.return_value = _result(rows)

    repo = YDBIncidentRepository()
    result = repo.get_stale_incidents()  # no args -> defaults

    returned_ids = {i.id for i in result}
    assert returned_ids == {"just-over", "accepted-just-over"}, (
        f"defaults must be (1800, 3600); got returned ids {returned_ids}"
    )


# ---------------------------------------------------------------------------
# 5-7. get_recent_nearby_incident behaviour
# ---------------------------------------------------------------------------

# Reference search point. Using ~100m and ~1500m offsets in latitude:
# 1 degree of latitude ~= 111_320 m, so 0.0009 deg ~= 100m, 0.0135 deg ~= 1500m.
SEARCH_LAT = 57.3000
SEARCH_LON = 44.6000
NEAR_LAT = SEARCH_LAT + 0.0009  # ~100m north
FAR_LAT = SEARCH_LAT + 0.0135  # ~1500m north


@patch("cloud.db.ydb_client.execute_query")
@patch("cloud.db.ydb_client.get_pool")
def test_get_recent_nearby_incident_finds_within_radius(
    mock_get_pool, mock_execute_query
):
    """Returns the pending incident inside radius_m, ignores the far one."""
    now = time.time()
    rows = [
        _row(
            id="near",
            status="pending",
            lat=NEAR_LAT,
            lon=SEARCH_LON,
            created_at=now - 30,
        ),
        _row(
            id="far",
            status="pending",
            lat=FAR_LAT,
            lon=SEARCH_LON,
            created_at=now - 30,
        ),
    ]
    mock_get_pool.return_value = _make_pool_mock()
    mock_execute_query.return_value = _result(rows)

    repo = YDBIncidentRepository()
    result = repo.get_recent_nearby_incident(
        lat=SEARCH_LAT, lon=SEARCH_LON, radius_m=500, max_age_s=300
    )

    assert isinstance(result, Incident)
    assert result.id == "near"


@patch("cloud.db.ydb_client.execute_query")
@patch("cloud.db.ydb_client.get_pool")
def test_get_recent_nearby_incident_returns_none_when_too_old(
    mock_get_pool, mock_execute_query
):
    """A spatially-close incident older than max_age_s must NOT match."""
    now = time.time()
    rows = [
        _row(
            id="stale-but-near",
            status="pending",
            lat=NEAR_LAT,
            lon=SEARCH_LON,
            created_at=now - 1000,  # > max_age_s=300
        ),
    ]
    mock_get_pool.return_value = _make_pool_mock()
    mock_execute_query.return_value = _result(rows)

    repo = YDBIncidentRepository()
    result = repo.get_recent_nearby_incident(
        lat=SEARCH_LAT, lon=SEARCH_LON, radius_m=500, max_age_s=300
    )

    assert result is None


@patch("cloud.db.ydb_client.execute_query")
@patch("cloud.db.ydb_client.get_pool")
def test_get_recent_nearby_incident_returns_none_when_no_pending(
    mock_get_pool, mock_execute_query
):
    """Only pending/accepted statuses count; resolved/false_alarm are skipped."""
    now = time.time()
    rows = [
        _row(
            id="nearby-but-resolved",
            status="resolved",
            lat=NEAR_LAT,
            lon=SEARCH_LON,
            created_at=now - 30,
        ),
    ]
    mock_get_pool.return_value = _make_pool_mock()
    mock_execute_query.return_value = _result(rows)

    repo = YDBIncidentRepository()
    result = repo.get_recent_nearby_incident(
        lat=SEARCH_LAT, lon=SEARCH_LON, radius_m=500, max_age_s=300
    )

    assert result is None
