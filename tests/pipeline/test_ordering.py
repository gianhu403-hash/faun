"""Tests for faun.ordering — sorting + gap detection."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from faun.ingest import AudioFileEntry, Manifest
from faun.ordering import (
    NORMAL_CYCLE_SECONDS,
    Gap,
    detect_gaps,
    sort_entries,
)


def _entry(
    trap_id: str,
    start_dt: dt.datetime | None,
    *,
    duration_s: float | None = 600.0,
    name: str = "f.wav",
) -> AudioFileEntry:
    return AudioFileEntry(
        path=Path(f"/tmp/{trap_id}/{name}"),
        trap_id=trap_id,
        start_dt=start_dt,
        lat=None,
        lon=None,
        meta={},
        duration_s=duration_s,
        sr=48000,
    )


def _t(minute: int, hour: int = 8) -> dt.datetime:
    return dt.datetime(2026, 1, 15, hour, minute, 0)


# ---------------------------------------------------------------------------
# sort_entries
# ---------------------------------------------------------------------------


class TestSortEntries:
    def test_sorts_by_time_within_trap(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(41), name="late.wav"),
                _entry("A1", _t(30), name="early.wav"),
            ],
        )
        out = sort_entries(m)
        assert [e.path.name for e in out.entries] == ["early.wav", "late.wav"]

    def test_sorts_by_trap_then_time(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A2", _t(0, hour=9)),
                _entry("A1", _t(41)),
                _entry("A1", _t(30)),
            ],
        )
        out = sort_entries(m)
        keys = [(e.trap_id, e.start_dt) for e in out.entries]
        assert keys == [
            ("A1", _t(30)),
            ("A1", _t(41)),
            ("A2", _t(0, hour=9)),
        ]

    def test_undated_entries_go_last_stable(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", None, name="undated1.wav"),
                _entry("A1", _t(30), name="dated.wav"),
                _entry("A1", None, name="undated2.wav"),
            ],
        )
        out = sort_entries(m)
        names = [e.path.name for e in out.entries]
        assert names == ["dated.wav", "undated1.wav", "undated2.wav"]

    def test_does_not_mutate_input(self):
        original = [
            _entry("A1", _t(41), name="late.wav"),
            _entry("A1", _t(30), name="early.wav"),
        ]
        m = Manifest(root=Path("/tmp"), entries=original)
        sort_entries(m)
        # Input order unchanged.
        assert [e.path.name for e in m.entries] == ["late.wav", "early.wav"]

    def test_empty_manifest(self):
        out = sort_entries(Manifest(root=Path("/tmp"), entries=[]))
        assert out.entries == []


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_normal_cycle_no_gap(self):
        # 10 min record + 1 min pause = next file starts at +11 min.
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=600.0),
                _entry("A1", _t(11), duration_s=600.0),
            ],
        )
        assert detect_gaps(m) == []

    def test_long_pause_flagged(self):
        # First file ends at 08:10; next starts at 09:00 -> 50 min gap.
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=600.0),
                _entry("A1", _t(0, hour=9), duration_s=600.0),
            ],
        )
        gaps = detect_gaps(m)
        assert len(gaps) == 1
        g = gaps[0]
        assert isinstance(g, Gap)
        assert g.trap_id == "A1"
        assert g.gap_start == _t(10)  # 08:00 + 600s
        assert g.gap_seconds == 50 * 60

    def test_gap_not_reported_across_traps(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=600.0),
                _entry("A2", _t(0, hour=12), duration_s=600.0),
            ],
        )
        assert detect_gaps(m) == []

    def test_unsorted_input_is_handled(self):
        # Provide out-of-order; detect_gaps sorts internally.
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0, hour=9), duration_s=600.0),
                _entry("A1", _t(0), duration_s=600.0),
            ],
        )
        gaps = detect_gaps(m)
        assert len(gaps) == 1
        assert gaps[0].gap_seconds == 50 * 60

    def test_undated_entries_ignored(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=600.0),
                _entry("A1", None, duration_s=600.0),
                _entry("A1", _t(11), duration_s=600.0),
            ],
        )
        # Undated breaks the chain but the two dated ones are a normal cycle.
        assert detect_gaps(m) == []

    def test_missing_duration_treated_as_zero(self):
        # Without duration, gap is measured from start-to-start.
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=None),
                _entry("A1", _t(5), duration_s=None),
            ],
        )
        gaps = detect_gaps(m)
        # 5 min start-to-start, threshold 120s -> flagged.
        assert len(gaps) == 1
        assert gaps[0].gap_seconds == 5 * 60

    def test_custom_threshold(self):
        m = Manifest(
            root=Path("/tmp"),
            entries=[
                _entry("A1", _t(0), duration_s=600.0),
                _entry("A1", _t(15), duration_s=600.0),  # 5 min pause
            ],
        )
        # Default threshold 120s -> flagged.
        assert len(detect_gaps(m)) == 1
        # Large threshold -> not flagged.
        assert detect_gaps(m, threshold_seconds=NORMAL_CYCLE_SECONDS) == []

    def test_empty_manifest(self):
        assert detect_gaps(Manifest(root=Path("/tmp"), entries=[])) == []
