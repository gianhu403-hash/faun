"""Tests for cloud.db.rangers — Ranger database and zone-based routing.

15 tests covering CRUD operations, zone queries, and edge cases.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Override DB path before importing the module
_tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
_tmp.close()
os.environ["RANGERS_DB_PATH"] = _tmp.name

from cloud.db.rangers import (
    Ranger,
    add_ranger,
    get_all_rangers,
    get_ranger_by_chat_id,
    get_rangers_for_location,
    init_db,
    _migrate_db,
    remove_ranger,
    set_active,
    update_zone,
)


@pytest.fixture(autouse=True)
def _clean_db():
    """Re-initialize DB before each test."""
    import sqlite3

    conn = sqlite3.connect(os.environ["RANGERS_DB_PATH"])
    conn.execute("DROP TABLE IF EXISTS rangers")
    conn.commit()
    conn.close()
    init_db()
    _migrate_db()
    yield


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestRangerCrud:
    def test_add_ranger(self) -> None:
        r = add_ranger("Иван", chat_id=12345)
        assert r.name == "Иван"
        assert r.chat_id == 12345
        assert r.active is True

    def test_add_ranger_with_zone(self) -> None:
        r = add_ranger(
            "Пётр",
            chat_id=99999,
            zone_lat_min=55.0,
            zone_lat_max=56.0,
            zone_lon_min=37.0,
            zone_lon_max=38.0,
        )
        assert r.zone_lat_min == 55.0
        assert r.zone_lon_max == 38.0

    def test_get_all_rangers(self) -> None:
        add_ranger("A", chat_id=1)
        add_ranger("B", chat_id=2)
        assert len(get_all_rangers()) == 2

    def test_get_ranger_by_chat_id(self) -> None:
        add_ranger("Тест", chat_id=42)
        r = get_ranger_by_chat_id(42)
        assert r is not None
        assert r.name == "Тест"

    def test_get_ranger_not_found(self) -> None:
        assert get_ranger_by_chat_id(99999) is None

    def test_remove_ranger(self) -> None:
        add_ranger("Удалить", chat_id=777)
        assert remove_ranger(777) is True
        assert get_ranger_by_chat_id(777) is None

    def test_remove_nonexistent(self) -> None:
        assert remove_ranger(99999) is False

    def test_set_active(self) -> None:
        add_ranger("Активный", chat_id=100)
        set_active(100, False)
        r = get_ranger_by_chat_id(100)
        assert r.active is False
        set_active(100, True)
        r = get_ranger_by_chat_id(100)
        assert r.active is True

    def test_update_zone(self) -> None:
        add_ranger("Зона", chat_id=200)
        update_zone(200, 55.0, 56.0, 37.0, 38.0)
        r = get_ranger_by_chat_id(200)
        assert r.zone_lat_min == 55.0
        assert r.zone_lat_max == 56.0


# ---------------------------------------------------------------------------
# Zone-based routing
# ---------------------------------------------------------------------------


class TestZoneRouting:
    def test_ranger_covers_location(self) -> None:
        add_ranger("Лесник1", chat_id=1, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        rangers = get_rangers_for_location(55.5, 37.5)
        assert len(rangers) == 1
        assert rangers[0].chat_id == 1

    def test_ranger_outside_zone(self) -> None:
        add_ranger("Далёкий", chat_id=2, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        rangers = get_rangers_for_location(60.0, 40.0)
        assert len(rangers) == 0

    def test_multiple_rangers_same_zone(self) -> None:
        add_ranger("Лесник1", chat_id=10, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        add_ranger("Лесник2", chat_id=20, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        rangers = get_rangers_for_location(55.5, 37.5)
        assert len(rangers) == 2

    def test_inactive_ranger_excluded(self) -> None:
        add_ranger("Выходной", chat_id=30, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        set_active(30, False)
        rangers = get_rangers_for_location(55.5, 37.5)
        assert len(rangers) == 0

    def test_overlapping_zones(self) -> None:
        """Two rangers with overlapping zones — both should get the alert."""
        add_ranger("Зона1", chat_id=40, zone_lat_min=55.0, zone_lat_max=56.0,
                    zone_lon_min=37.0, zone_lon_max=38.0)
        add_ranger("Зона2", chat_id=50, zone_lat_min=55.5, zone_lat_max=56.5,
                    zone_lon_min=37.5, zone_lon_max=38.5)
        # Point in overlap area
        rangers = get_rangers_for_location(55.7, 37.7)
        assert len(rangers) == 2

    def test_unique_chat_id_constraint(self) -> None:
        """Cannot add two rangers with the same chat_id."""
        add_ranger("Один", chat_id=999)
        with pytest.raises(Exception):
            add_ranger("Два", chat_id=999)
