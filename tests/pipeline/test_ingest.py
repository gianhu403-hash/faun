"""Tests for faun.ingest — directory scanning + info.txt / filename parsing."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from faun.ingest import AudioFileEntry, Manifest, scan
from faun.ingest.files import (
    _parse_filename_timestamp,
    parse_info_txt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, seconds: float = 0.1, sr: int = 48000, channels: int = 2):
    """Write a tiny stereo WAV (default 48k, matching trap recordings)."""
    n = int(sr * seconds)
    data = np.zeros((n, channels), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)


_INFO_HEADER = (
    "date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel\n"
)


def _make_trap(
    root: Path,
    trap_id: str,
    files: list[tuple[str, str, str]],
    *,
    long: str = "44.123",
    lat: str = "57.456",
    with_info: bool = True,
    sr: int = 48000,
):
    """Create a trap folder with WAVs and (optionally) an info.txt.

    ``files`` is a list of (filename, date, time) tuples.
    """
    folder = root / trap_id
    folder.mkdir(parents=True, exist_ok=True)
    for fname, _, _ in files:
        _write_wav(folder / fname, sr=sr)
    if with_info:
        lines = [_INFO_HEADER]
        for fname, date_s, time_s in files:
            lines.append(
                f"{date_s},{time_s},{long},{lat},85,12.5,60,{fname},{sr},auto,stereo\n"
            )
        (folder / "info.txt").write_text("".join(lines), encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# Filename timestamp parsing
# ---------------------------------------------------------------------------


class TestFilenameTimestamp:
    def test_underscore_pattern(self):
        ts = _parse_filename_timestamp("20260115_083000.wav")
        assert ts == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_compact_pattern(self):
        ts = _parse_filename_timestamp("20260115083000.wav")
        assert ts == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_dashed_pattern(self):
        ts = _parse_filename_timestamp("2026-01-15_08-30-00.wav")
        assert ts == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_prefixed_pattern(self):
        ts = _parse_filename_timestamp("REC_20260115_083000.wav")
        assert ts == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_no_timestamp(self):
        assert _parse_filename_timestamp("recording.wav") is None

    def test_invalid_date_skipped(self):
        # Month 13 is not a real date.
        assert _parse_filename_timestamp("20261315_083000.wav") is None


# ---------------------------------------------------------------------------
# info.txt parsing
# ---------------------------------------------------------------------------


class TestParseInfoTxt:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "info.txt"
        p.write_text(
            _INFO_HEADER
            + "2026-01-15,08:30:00,44.1,57.4,85,12.5,60,a.wav,48000,auto,stereo\n",
            encoding="utf-8",
        )
        rows = parse_info_txt(p)
        assert "a.wav" in rows
        assert rows["a.wav"]["long"] == "44.1"
        assert rows["a.wav"]["lat"] == "57.4"

    def test_header_aliases(self, tmp_path: Path):
        p = tmp_path / "info.txt"
        p.write_text(
            "date,time,longitude,latitude,batt,temperature,humi,"
            "file,samplerate,gain,channels\n"
            "2026-01-15,08:30:00,44.1,57.4,85,12.5,60,a.wav,48000,9,2\n",
            encoding="utf-8",
        )
        rows = parse_info_txt(p)
        assert rows["a.wav"]["long"] == "44.1"
        assert rows["a.wav"]["sample_rate"] == "48000"

    def test_semicolon_delimiter(self, tmp_path: Path):
        p = tmp_path / "info.txt"
        p.write_text(
            _INFO_HEADER.replace(",", ";")
            + "2026-01-15;08:30:00;44.1;57.4;85;12.5;60;a.wav;48000;auto;stereo\n",
            encoding="utf-8",
        )
        rows = parse_info_txt(p)
        assert "a.wav" in rows
        assert rows["a.wav"]["lat"] == "57.4"

    def test_no_header_assumes_canonical(self, tmp_path: Path):
        p = tmp_path / "info.txt"
        p.write_text(
            "2026-01-15,08:30:00,44.1,57.4,85,12.5,60,a.wav,48000,auto,stereo\n",
            encoding="utf-8",
        )
        rows = parse_info_txt(p)
        assert "a.wav" in rows
        assert rows["a.wav"]["long"] == "44.1"

    def test_broken_row_skipped_with_warning(self, tmp_path: Path, caplog):
        p = tmp_path / "info.txt"
        p.write_text(
            _INFO_HEADER
            + "2026-01-15,08:30:00,44.1,57.4,85,12.5,60,good.wav,48000,auto,stereo\n"
            + "garbage,row\n",  # too few columns
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            rows = parse_info_txt(p)
        assert "good.wav" in rows
        assert len(rows) == 1
        assert any("malformed" in r.message.lower() for r in caplog.records)

    def test_extra_trailing_column_tolerated(self, tmp_path: Path):
        # Roman mentioned an empty "extra" graph; a missing trailing column
        # should still parse.
        p = tmp_path / "info.txt"
        p.write_text(
            _INFO_HEADER
            + "2026-01-15,08:30:00,44.1,57.4,85,12.5,60,a.wav,48000,auto\n",
            encoding="utf-8",
        )
        rows = parse_info_txt(p)
        assert "a.wav" in rows


# ---------------------------------------------------------------------------
# scan() — full dataset
# ---------------------------------------------------------------------------


class TestScan:
    def test_single_trap(self, tmp_path: Path):
        _make_trap(
            tmp_path,
            "A1",
            [
                ("20260115_083000.wav", "2026-01-15", "08:30:00"),
                ("20260115_084100.wav", "2026-01-15", "08:41:00"),
            ],
        )
        manifest = scan(tmp_path)
        assert isinstance(manifest, Manifest)
        assert len(manifest) == 2
        e = manifest.entries[0]
        assert isinstance(e, AudioFileEntry)
        assert e.trap_id == "A1"
        assert e.start_dt == dt.datetime(2026, 1, 15, 8, 30, 0)
        assert e.lon == 44.123
        assert e.lat == 57.456
        assert e.sr == 48000
        assert e.duration_s is not None and e.duration_s > 0
        assert e.meta["temp"] == 12.5
        assert e.meta["channel"] == "stereo"

    def test_multiple_traps(self, tmp_path: Path):
        _make_trap(tmp_path, "A1", [("20260115_083000.wav", "2026-01-15", "08:30:00")])
        _make_trap(tmp_path, "A2", [("20260115_090000.wav", "2026-01-15", "09:00:00")])
        manifest = scan(tmp_path)
        assert len(manifest) == 2
        assert set(manifest.trap_ids) == {"A1", "A2"}

    def test_timestamp_falls_back_to_info(self, tmp_path: Path):
        # Filename has no parseable stamp -> use info.txt date/time.
        _make_trap(
            tmp_path,
            "A1",
            [("recording.wav", "2026-01-15", "08:30:00")],
        )
        manifest = scan(tmp_path)
        assert manifest.entries[0].start_dt == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_no_timestamp_anywhere(self, tmp_path: Path):
        folder = tmp_path / "A1"
        _write_wav(folder / "recording.wav")
        # info.txt without date/time
        (folder / "info.txt").write_text(
            _INFO_HEADER + ",,44.1,57.4,85,12.5,60,recording.wav,48000,auto,stereo\n",
            encoding="utf-8",
        )
        manifest = scan(tmp_path)
        assert manifest.entries[0].start_dt is None
        # Coordinates still recovered.
        assert manifest.entries[0].lon == 44.1

    def test_flat_directory_without_info(self, tmp_path: Path):
        # WAVs directly under root, no info.txt -> single trap named after root.
        _write_wav(tmp_path / "20260115_083000.wav")
        _write_wav(tmp_path / "20260115_084100.wav")
        manifest = scan(tmp_path)
        assert len(manifest) == 2
        assert manifest.entries[0].trap_id == tmp_path.name
        assert manifest.entries[0].meta == {}
        assert manifest.entries[0].start_dt == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_trap_without_info(self, tmp_path: Path):
        _make_trap(
            tmp_path,
            "A1",
            [("20260115_083000.wav", "", "")],
            with_info=False,
        )
        manifest = scan(tmp_path)
        assert len(manifest) == 1
        assert manifest.entries[0].trap_id == "A1"
        assert manifest.entries[0].meta == {}
        # Timestamp still from filename.
        assert manifest.entries[0].start_dt == dt.datetime(2026, 1, 15, 8, 30, 0)

    def test_empty_directory(self, tmp_path: Path):
        manifest = scan(tmp_path)
        assert len(manifest) == 0

    def test_missing_path_raises(self, tmp_path: Path):
        try:
            scan(tmp_path / "does_not_exist")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")

    def test_file_instead_of_dir_raises(self, tmp_path: Path):
        f = tmp_path / "file.wav"
        _write_wav(f)
        try:
            scan(f)
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("expected NotADirectoryError")

    def test_duration_and_sr_from_header(self, tmp_path: Path):
        _make_trap(
            tmp_path,
            "A1",
            [("20260115_083000.wav", "2026-01-15", "08:30:00")],
            sr=48000,
        )
        e = scan(tmp_path).entries[0]
        assert e.sr == 48000
        # 0.1 s default duration.
        assert abs(e.duration_s - 0.1) < 0.05
