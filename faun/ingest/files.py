"""Directory scanning and info.txt / filename parsing for the Faun pipeline.

A trap dataset looks like::

    dataset/
        A1/
            info.txt
            20260115_083000.wav
            20260115_084100.wav
        A2/
            info.txt
            ...

``info.txt`` is CSV (comma separated, ``.txt`` extension) with columns::

    date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel

(see docs/business/meetings MoM — Roman's description of the trap output).
Note: ``long`` precedes ``lat`` in the file, i.e. longitude then latitude.

The timestamp is read primarily from the WAV filename (typical recorder
patterns like ``YYYYMMDD_HHMMSS``); if the filename has no parseable stamp we
fall back to the ``date``/``time`` columns of ``info.txt``; otherwise
``start_dt`` is ``None``.

Robustness: header variations and the field delimiter are auto-detected,
broken rows are skipped with a logged warning, and audio duration / sample
rate are read via ``soundfile.info`` (header only, the file is not loaded).
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

WAV_SUFFIXES = (".wav", ".wave")
INFO_FILENAME = "info.txt"

# Canonical info.txt header (per the frozen interface contract).
_CANONICAL_COLUMNS = (
    "date",
    "time",
    "long",
    "lat",
    "battery",
    "temp",
    "humidity",
    "filename",
    "sample_rate",
    "gain",
    "channel",
)

# Header aliases -> canonical name. Lower-cased, stripped before lookup.
_HEADER_ALIASES = {
    "date": "date",
    "time": "time",
    "long": "long",
    "lon": "long",
    "lng": "long",
    "longitude": "long",
    "lat": "lat",
    "latitude": "lat",
    "battery": "battery",
    "batt": "battery",
    "temp": "temp",
    "temperature": "temp",
    "humidity": "humidity",
    "humi": "humidity",
    "hum": "humidity",
    "filename": "filename",
    "file": "filename",
    "name": "filename",
    "sample_rate": "sample_rate",
    "samplerate": "sample_rate",
    "sr": "sample_rate",
    "rate": "sample_rate",
    "gain": "gain",
    "channel": "channel",
    "channels": "channel",
}

# Filename timestamp patterns, tried in order. Each yields a datetime.
#   YYYYMMDD_HHMMSS / YYYYMMDDHHMMSS / YYYY-MM-DD_HH-MM-SS / YYYYMMDD-HHMMSS
_FILENAME_TS_PATTERNS = (
    re.compile(
        r"(?P<Y>\d{4})[-_]?(?P<m>\d{2})[-_]?(?P<d>\d{2})"
        r"[-_T ]"
        r"(?P<H>\d{2})[-_:]?(?P<M>\d{2})[-_:]?(?P<S>\d{2})"
    ),
    # Compact with no separator at all: YYYYMMDDHHMMSS
    re.compile(
        r"(?P<Y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"
        r"(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"
    ),
)


@dataclass
class AudioFileEntry:
    """A single audio recording plus the metadata we managed to recover."""

    path: Path
    trap_id: str
    start_dt: _dt.datetime | None
    lat: float | None
    lon: float | None
    meta: dict = field(default_factory=dict)
    duration_s: float | None = None
    sr: int | None = None


@dataclass
class Manifest:
    """An ordered collection of :class:`AudioFileEntry` for a scanned root."""

    root: Path
    entries: list[AudioFileEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def trap_ids(self) -> list[str]:
        """Distinct trap ids in first-seen order."""
        seen: list[str] = []
        for e in self.entries:
            if e.trap_id not in seen:
                seen.append(e.trap_id)
        return seen


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_filename_timestamp(name: str) -> _dt.datetime | None:
    """Extract a datetime from a recorder filename, or ``None``."""
    for pattern in _FILENAME_TS_PATTERNS:
        m = pattern.search(name)
        if not m:
            continue
        try:
            return _dt.datetime(
                int(m.group("Y")),
                int(m.group("m")),
                int(m.group("d")),
                int(m.group("H")),
                int(m.group("M")),
                int(m.group("S")),
            )
        except ValueError:
            # Matched digits but not a real date (e.g. month 13) — keep trying.
            continue
    return None


def _parse_info_timestamp(
    date_s: str | None, time_s: str | None
) -> _dt.datetime | None:
    """Combine info.txt ``date`` + ``time`` columns into a datetime."""
    if not date_s:
        return None
    date_s = date_s.strip()
    time_s = (time_s or "").strip()
    candidate = f"{date_s} {time_s}".strip()
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d %H%M%S",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%Y-%m-%d",
        "%d.%m.%Y",
    )
    for fmt in fmts:
        try:
            return _dt.datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = (
        value.strip().replace(",", ".")
        if value.strip().count(",") and "." not in value
        else value.strip()
    )
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    f = _to_float(value)
    if f is None:
        return None
    return int(f)


def _normalize_header(field_name: str) -> str:
    return _HEADER_ALIASES.get(field_name.strip().lower(), field_name.strip().lower())


def _looks_like_header(fields: list[str]) -> bool:
    """True if the row resembles a header (contains known column names)."""
    normalized = {_normalize_header(f) for f in fields}
    return len(normalized & set(_CANONICAL_COLUMNS)) >= 2


def parse_info_txt(info_path: Path) -> dict[str, dict]:
    """Parse an ``info.txt`` into ``{filename: metadata-dict}``.

    Header variations and the delimiter are auto-detected. Broken rows
    (wrong column count, unparseable) are skipped with a warning. If the file
    has no header at all, the canonical column order is assumed.
    """
    rows: dict[str, dict] = {}
    try:
        text = info_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        logger.warning("ingest: cannot read %s: %s", info_path, exc)
        return rows
    except UnicodeDecodeError:
        text = info_path.read_text(encoding="latin-1")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        logger.warning("ingest: empty info.txt %s", info_path)
        return rows

    # Detect delimiter from the first non-empty line.
    sample = lines[0]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    reader = csv.reader(lines, delimiter=delimiter)
    records = list(reader)
    if not records:
        return rows

    first = [c.strip() for c in records[0]]
    if _looks_like_header(first):
        columns = [_normalize_header(c) for c in first]
        data_rows = records[1:]
    else:
        columns = list(_CANONICAL_COLUMNS)
        data_rows = records

    for raw in data_rows:
        cells = [c.strip() for c in raw]
        if not any(cells):
            continue
        if len(cells) < len(columns) - 1:
            # Tolerate a single missing trailing column (often the empty
            # "extra" graph Roman mentioned), otherwise skip as broken.
            logger.warning("ingest: skipping malformed row in %s: %r", info_path, raw)
            continue
        record = {columns[i]: cells[i] for i in range(min(len(columns), len(cells)))}
        fname = record.get("filename", "").strip()
        if not fname:
            logger.warning("ingest: row without filename in %s: %r", info_path, raw)
            continue
        rows[fname] = record
    return rows


# ---------------------------------------------------------------------------
# Audio header probing
# ---------------------------------------------------------------------------


def _probe_audio(path: Path) -> tuple[float | None, int | None]:
    """Read duration (s) and sample rate from the WAV header only."""
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises RuntimeError / LibsndfileError
        logger.warning("ingest: cannot probe audio %s: %s", path, exc)
        return None, None
    duration = info.frames / info.samplerate if info.samplerate else None
    return duration, info.samplerate


# ---------------------------------------------------------------------------
# Entry building
# ---------------------------------------------------------------------------


def _build_entry(wav_path: Path, trap_id: str, info_row: dict | None) -> AudioFileEntry:
    info_row = info_row or {}

    start_dt = _parse_filename_timestamp(wav_path.name)
    if start_dt is None:
        start_dt = _parse_info_timestamp(info_row.get("date"), info_row.get("time"))

    lon = _to_float(info_row.get("long"))
    lat = _to_float(info_row.get("lat"))

    meta: dict = {}
    if info_row:
        meta = {
            "battery": _to_float(info_row.get("battery")),
            "temp": _to_float(info_row.get("temp")),
            "humidity": _to_float(info_row.get("humidity")),
            "sample_rate": _to_int(info_row.get("sample_rate")),
            "gain": info_row.get("gain", "").strip() or None,
            "channel": info_row.get("channel", "").strip() or None,
        }
        # Drop keys whose value is None so meta stays clean.
        meta = {k: v for k, v in meta.items() if v is not None}

    duration_s, sr = _probe_audio(wav_path)
    # Prefer the actually-decoded sample rate; fall back to info.txt.
    if sr is None and "sample_rate" in meta:
        sr = meta["sample_rate"]

    return AudioFileEntry(
        path=wav_path,
        trap_id=trap_id,
        start_dt=start_dt,
        lat=lat,
        lon=lon,
        meta=meta,
        duration_s=duration_s,
        sr=sr,
    )


def _iter_wavs(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in WAV_SUFFIXES
    )


def _scan_trap_folder(folder: Path) -> list[AudioFileEntry]:
    trap_id = folder.name
    info_path = folder / INFO_FILENAME
    info_map: dict[str, dict] = {}
    if info_path.is_file():
        info_map = parse_info_txt(info_path)

    entries: list[AudioFileEntry] = []
    for wav in _iter_wavs(folder):
        info_row = info_map.get(wav.name)
        entries.append(_build_entry(wav, trap_id, info_row))
    return entries


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan(path: Path) -> Manifest:
    """Scan ``path`` and return a :class:`Manifest`.

    ``path`` may be either:

    * a dataset root containing trap subfolders (``A1``, ``A2``, …), each
      optionally with an ``info.txt``; or
    * a flat folder of WAV files (treated as a single trap named after the
      folder, with no info.txt metadata).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ingest.scan: path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"ingest.scan: not a directory: {path}")

    subdirs = sorted(p for p in path.iterdir() if p.is_dir())
    direct_wavs = _iter_wavs(path)

    entries: list[AudioFileEntry] = []

    if subdirs:
        for folder in subdirs:
            entries.extend(_scan_trap_folder(folder))

    # WAVs sitting directly in the root form a trap named after the root.
    if direct_wavs:
        info_path = path / INFO_FILENAME
        info_map = parse_info_txt(info_path) if info_path.is_file() else {}
        for wav in direct_wavs:
            entries.append(_build_entry(wav, path.name, info_map.get(wav.name)))

    if not entries:
        logger.warning("ingest: no WAV files found under %s", path)

    return Manifest(root=path, entries=entries)
