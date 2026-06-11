"""Ordering: chronological ordering of manifest entries (Phase 2).

Public API::

    from faun.ordering import sort_entries, detect_gaps, Gap

``sort_entries(manifest)`` returns a new :class:`~faun.ingest.Manifest` whose
entries are ordered by ``(trap_id, start_dt)``. Entries without a timestamp
are placed at the end of their trap group, preserving their original relative
order (stable sort).

``detect_gaps(manifest)`` reports pauses between consecutive recordings of a
trap that exceed the normal duty cycle (a ~10 min recording followed by a
~1 min pause). A larger pause usually means the trap was off, moved, or files
are missing — useful for flagging timeline holes before classification.
"""

from __future__ import annotations

from dataclasses import dataclass

from faun.ingest import AudioFileEntry, Manifest

__all__ = ["sort_entries", "detect_gaps", "Gap", "NORMAL_CYCLE_SECONDS"]

# A normal duty cycle: ~10 min recording + ~1 min pause. We treat the END of
# one recording to the START of the next as the gap; anything beyond the
# expected pause (with margin) is a real gap. Expected pause ~= 60 s; we use a
# generous threshold so jitter and rounding do not produce false positives.
RECORDING_SECONDS = 10 * 60
PAUSE_SECONDS = 60
NORMAL_CYCLE_SECONDS = RECORDING_SECONDS + PAUSE_SECONDS  # 660 s, one full cycle
# Tolerated pause between the end of one file and the start of the next.
_GAP_THRESHOLD_SECONDS = 2 * PAUSE_SECONDS  # 120 s — twice the nominal pause


@dataclass
class Gap:
    """A timeline hole in a single trap's recording sequence."""

    trap_id: str
    gap_start: object  # datetime — end of the recording preceding the gap
    gap_seconds: float


def _sort_key(entry: AudioFileEntry):
    """Sort by trap, then by time; undated entries sort after dated ones.

    Python sorts are stable, so equal keys keep their original order — which
    is exactly what we want for undated files within a trap.
    """
    has_dt = entry.start_dt is not None
    # ``not has_dt`` -> dated (False) before undated (True).
    # ``start_dt`` only compared among dated entries (same trap, both dated).
    return (
        entry.trap_id,
        not has_dt,
        entry.start_dt if has_dt else None,
    )


def sort_entries(manifest: Manifest) -> Manifest:
    """Return a new manifest sorted by ``(trap_id, start_dt)``.

    The input manifest is not mutated. Undated entries land at the end of
    their trap group in stable original order.
    """
    ordered = sorted(manifest.entries, key=_sort_key)
    return Manifest(root=manifest.root, entries=ordered)


def detect_gaps(
    manifest: Manifest, threshold_seconds: float = _GAP_THRESHOLD_SECONDS
) -> list[Gap]:
    """Find pauses larger than the normal duty cycle, per trap.

    A gap is measured from the *end* of one recording (start + duration) to
    the *start* of the next. Only consecutive dated entries with a known
    duration contribute. The manifest is sorted internally first, so the
    caller need not pre-sort.

    Returns a list of :class:`Gap`, ordered by trap then time.
    """
    ordered = sort_entries(manifest)

    gaps: list[Gap] = []
    prev: AudioFileEntry | None = None
    for entry in ordered.entries:
        if entry.start_dt is None:
            # Undated tail of a trap — nothing meaningful to compare.
            prev = None
            continue
        if prev is None or prev.trap_id != entry.trap_id:
            prev = entry
            continue

        prev_duration = prev.duration_s if prev.duration_s is not None else 0.0
        prev_end = prev.start_dt + _seconds(prev_duration)
        delta = (entry.start_dt - prev_end).total_seconds()
        if delta > threshold_seconds:
            gaps.append(
                Gap(
                    trap_id=entry.trap_id,
                    gap_start=prev_end,
                    gap_seconds=delta,
                )
            )
        prev = entry

    return gaps


def _seconds(value: float):
    import datetime as _dt

    return _dt.timedelta(seconds=value)
