"""TDD tests for random demo event location generation within Varnavino boundary."""

import math
import random

import pytest


# ---------------------------------------------------------------------------
# Tests for random_point_in_boundary()
# ---------------------------------------------------------------------------


class TestRandomPointInBoundary:
    def test_all_points_inside_polygon(self):
        """50 generated points must all be inside the boundary polygon."""
        from cloud.db.microphones import _point_in_polygon, random_point_in_boundary

        for _ in range(50):
            lat, lon = random_point_in_boundary()
            assert _point_in_polygon(lat, lon), f"Point ({lat}, {lon}) outside polygon"

    def test_spread_across_territory(self):
        """200 points must span at least 50% of lat and lon range (not clustered)."""
        from cloud.db.microphones import (
            LAT_MAX,
            LAT_MIN,
            LON_MAX,
            LON_MIN,
            random_point_in_boundary,
        )

        lats, lons = [], []
        for _ in range(200):
            lat, lon = random_point_in_boundary()
            lats.append(lat)
            lons.append(lon)

        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        lat_range = LAT_MAX - LAT_MIN
        lon_range = LON_MAX - LON_MIN

        assert lat_span >= 0.5 * lat_range, (
            f"Lat span {lat_span:.4f} < 50% of range {lat_range:.4f}"
        )
        assert lon_span >= 0.5 * lon_range, (
            f"Lon span {lon_span:.4f} < 50% of range {lon_range:.4f}"
        )

    def test_deterministic_with_seed(self):
        """Same seed produces same point."""
        from cloud.db.microphones import random_point_in_boundary

        random.seed(42)
        p1 = random_point_in_boundary()
        random.seed(42)
        p2 = random_point_in_boundary()
        assert p1 == p2

    def test_fallback_when_polygon_empty(self, monkeypatch):
        """With empty polygon, falls back to bbox — point still in bbox."""
        import cloud.db.microphones as mic_mod

        monkeypatch.setattr(mic_mod, "_BOUNDARY_POLYGON", [])

        from cloud.db.microphones import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

        for _ in range(10):
            lat, lon = mic_mod.random_point_in_boundary()
            assert LAT_MIN <= lat <= LAT_MAX
            assert LON_MIN <= lon <= LON_MAX


# ---------------------------------------------------------------------------
# Tests for get_nearest_online()
# ---------------------------------------------------------------------------


class TestGetNearestOnline:
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path, monkeypatch):
        """Use a temp SQLite DB for each test."""
        monkeypatch.delenv("YDB_ENDPOINT", raising=False)
        db_file = tmp_path / "mics.db"
        monkeypatch.setattr("cloud.db.microphones._db_path", lambda: str(db_file))
        from cloud.db.microphones import init_db

        init_db()

    def _insert_mic(self, mic_uid, lat, lon, status="online"):
        from cloud.db.microphones import _get_conn

        conn = _get_conn()
        conn.execute(
            "INSERT INTO microphones (mic_uid, lat, lon, zone_type, sub_district, status, battery_pct, district_slug, installed_at) "
            "VALUES (?, ?, ?, 'exploitation', 'Варнавинское', ?, 100.0, 'varnavino', '2025-01-01T00:00:00')",
            (mic_uid, lat, lon, status),
        )
        conn.commit()
        conn.close()

    def test_returns_closest_3(self):
        """With 5 mics, returns the 3 closest to the query point."""
        from cloud.db.microphones import get_nearest_online

        # Query point: 57.5, 44.8
        # Distances (approx): mic1 closest, mic2 next, ... mic5 farthest
        self._insert_mic("m1", 57.50, 44.80)  # ~0 km
        self._insert_mic("m2", 57.51, 44.81)  # ~1.3 km
        self._insert_mic("m3", 57.52, 44.82)  # ~2.6 km
        self._insert_mic("m4", 57.60, 44.90)  # ~12.5 km
        self._insert_mic("m5", 57.70, 45.00)  # ~25 km

        result = get_nearest_online(57.50, 44.80, n=3)
        uids = [m.mic_uid for m in result]

        assert len(result) == 3
        assert uids == ["m1", "m2", "m3"]

    def test_fewer_than_n(self):
        """With only 2 online mics, returns 2 without error."""
        from cloud.db.microphones import get_nearest_online

        self._insert_mic("m1", 57.50, 44.80)
        self._insert_mic("m2", 57.51, 44.81)

        result = get_nearest_online(57.50, 44.80, n=3)
        assert len(result) == 2

    def test_empty_db(self):
        """With no online mics, returns empty list."""
        from cloud.db.microphones import get_nearest_online

        result = get_nearest_online(57.50, 44.80, n=3)
        assert result == []
