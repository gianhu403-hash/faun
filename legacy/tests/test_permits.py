"""Tests for cloud.db.permits — logging permit database.

Covers CRUD operations, geographic lookup, date-based validity,
and edge cases (expired, future, overlapping permits).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cloud.db.permits import (
    Permit,
    add_permit,
    get_all_permits,
    get_permits_for_location,
    has_valid_permit,
    init_db,
    remove_permit,
)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite database."""
    db_file = str(tmp_path / "permits_test.sqlite")
    monkeypatch.setenv("PERMITS_DB_PATH", db_file)
    init_db()


# Convenience dates
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
TOMORROW = TODAY + timedelta(days=1)
LAST_MONTH = TODAY - timedelta(days=30)
NEXT_MONTH = TODAY + timedelta(days=30)
LAST_YEAR = TODAY - timedelta(days=365)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestPermitCrud:
    def test_add_permit(self) -> None:
        p = add_permit(
            zone_lat_min=55.0, zone_lat_max=56.0,
            zone_lon_min=37.0, zone_lon_max=38.0,
            valid_from=TODAY, valid_until=NEXT_MONTH,
            description="Делянка №12",
        )
        assert isinstance(p, Permit)
        assert p.id is not None
        assert p.description == "Делянка №12"
        assert p.valid_from == TODAY
        assert p.valid_until == NEXT_MONTH

    def test_get_all_permits(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        add_permit(57.0, 58.0, 44.0, 45.0, TODAY, NEXT_MONTH)
        assert len(get_all_permits()) == 2

    def test_remove_permit(self) -> None:
        p = add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        assert remove_permit(p.id) is True
        assert len(get_all_permits()) == 0

    def test_remove_nonexistent(self) -> None:
        assert remove_permit(9999) is False

    def test_empty_db(self) -> None:
        assert get_all_permits() == []


# ---------------------------------------------------------------------------
# Geographic lookup
# ---------------------------------------------------------------------------


class TestGeographicLookup:
    def test_point_inside_zone(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        assert has_valid_permit(55.5, 37.5) is True

    def test_point_outside_zone(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        assert has_valid_permit(60.0, 37.5) is False

    def test_point_on_boundary(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        assert has_valid_permit(55.0, 37.0) is True
        assert has_valid_permit(56.0, 38.0) is True

    def test_no_permits_returns_false(self) -> None:
        assert has_valid_permit(55.5, 37.5) is False

    def test_multiple_permits_same_area(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH, "Билет 1")
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH, "Билет 2")
        permits = get_permits_for_location(55.5, 37.5)
        assert len(permits) == 2

    def test_overlapping_zones(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH)
        add_permit(55.5, 56.5, 37.5, 38.5, TODAY, NEXT_MONTH)
        # Point in overlap area
        assert has_valid_permit(55.7, 37.7) is True
        permits = get_permits_for_location(55.7, 37.7)
        assert len(permits) == 2


# ---------------------------------------------------------------------------
# Date-based validity
# ---------------------------------------------------------------------------


class TestDateValidity:
    def test_current_permit_valid(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, LAST_MONTH, NEXT_MONTH)
        assert has_valid_permit(55.5, 37.5) is True

    def test_expired_permit_invalid(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, LAST_YEAR, YESTERDAY)
        assert has_valid_permit(55.5, 37.5) is False

    def test_future_permit_invalid(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TOMORROW, NEXT_MONTH)
        assert has_valid_permit(55.5, 37.5) is False

    def test_permit_valid_on_exact_dates(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, TODAY)
        assert has_valid_permit(55.5, 37.5, on_date=TODAY) is True

    def test_custom_date_check(self) -> None:
        add_permit(55.0, 56.0, 37.0, 38.0, LAST_YEAR, YESTERDAY)
        # Was valid last year
        check_date = LAST_YEAR + timedelta(days=10)
        assert has_valid_permit(55.5, 37.5, on_date=check_date) is True
        # Not valid today
        assert has_valid_permit(55.5, 37.5, on_date=TODAY) is False

    def test_expired_and_active_same_zone(self) -> None:
        """Expired permit should not block; active permit should match."""
        add_permit(55.0, 56.0, 37.0, 38.0, LAST_YEAR, YESTERDAY, "Истёк")
        add_permit(55.0, 56.0, 37.0, 38.0, TODAY, NEXT_MONTH, "Действует")
        assert has_valid_permit(55.5, 37.5) is True
        permits = get_permits_for_location(55.5, 37.5)
        assert len(permits) == 1
        assert permits[0].description == "Действует"
