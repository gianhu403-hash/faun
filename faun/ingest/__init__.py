"""Ingest: scan trap folders + info.txt -> Manifest (Phase 2).

Public API::

    from faun.ingest import scan, Manifest, AudioFileEntry

``scan(path)`` walks a directory of trap folders (A1..A5), parses the
per-folder ``info.txt`` (CSV metadata) and the timestamp embedded in each
WAV filename, and returns a :class:`Manifest`. A flat directory of WAV files
without ``info.txt`` is also supported (metadata empty, ``trap_id`` from the
folder name).

Contract (faun/INTERFACES.md, FROZEN)::

    scan(path: Path) -> Manifest
    AudioFileEntry(path, trap_id, start_dt, lat, lon, meta, duration_s, sr)
    info.txt CSV columns:
        date,time,long,lat,battery,temp,humidity,filename,
        sample_rate,gain,channel
"""

from __future__ import annotations

from .files import AudioFileEntry, Manifest, scan

__all__ = ["AudioFileEntry", "Manifest", "scan"]
