"""Output: ``CsvWriter`` for species results + sidecar trap metadata (Phase 2).

The results CSV column order is frozen by ``faun/INTERFACES.md``::

    track, start_sec, duration_sec, species, probability

Alongside ``results.csv`` a ``results_meta.json`` sidecar carries per-trap
provenance (trap id, coordinates, source files, pipeline version) so the CSV
stays a flat, tool-friendly table while the metadata travels with it.

stdlib only — no heavy imports here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Iterable, Sequence

__all__ = [
    "PIPELINE_VERSION",
    "ResultRow",
    "TrapMeta",
    "CsvWriter",
    "COLUMNS",
]

# Pipeline schema/version stamped into the sidecar. Bump on breaking changes
# to the results.csv layout or sidecar shape.
PIPELINE_VERSION = "2.0"

# Frozen column order — see faun/INTERFACES.md. Do not reorder or rename.
COLUMNS: tuple[str, ...] = (
    "track",
    "start_sec",
    "duration_sec",
    "species",
    "probability",
)

# Rounding precision (see task contract): 2 decimals for seconds, 4 for probability.
_SEC_DIGITS = 2
_PROB_DIGITS = 4


@dataclass(frozen=True)
class ResultRow:
    """One classified segment, ready to serialise into results.csv.

    ``track`` is the source identifier (e.g. WAV filename or trap/track key);
    ``start_sec``/``duration_sec`` locate the segment on that track's timeline.
    """

    track: str
    start_sec: float
    duration_sec: float
    species: str
    probability: float

    def as_csv_dict(self) -> dict[str, object]:
        """Return a dict with values rounded for CSV serialisation."""
        return {
            "track": self.track,
            "start_sec": round(float(self.start_sec), _SEC_DIGITS),
            "duration_sec": round(float(self.duration_sec), _SEC_DIGITS),
            "species": self.species,
            "probability": round(float(self.probability), _PROB_DIGITS),
        }


@dataclass
class TrapMeta:
    """Sidecar provenance for a results.csv produced from one trap's folder.

    Fields mirror what ingest derives from ``info.txt`` (date,time,long,lat,
    battery,temp,humidity,filename,sample_rate,gain,channel) plus job-level
    bookkeeping. All optional except ``trap_id`` so callers can fill what they
    have without breaking the schema.
    """

    trap_id: str
    lat: float | None = None
    lon: float | None = None
    files: list[str] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view of the metadata."""
        data = asdict(self)
        # Flatten ``extra`` into the top level so consumers see one object,
        # but never let it shadow the canonical keys.
        extra = data.pop("extra")
        for key, value in extra.items():
            data.setdefault(key, value)
        return data


def _coerce_row(row: ResultRow | dict[str, object] | Sequence[object]) -> ResultRow:
    """Accept a ResultRow, a mapping, or a 5-tuple in canonical column order."""
    if isinstance(row, ResultRow):
        return row
    if isinstance(row, dict):
        return ResultRow(
            track=str(row["track"]),
            start_sec=float(row["start_sec"]),
            duration_sec=float(row["duration_sec"]),
            species=str(row["species"]),
            probability=float(row["probability"]),
        )
    seq = list(row)
    if len(seq) != len(COLUMNS):
        raise ValueError(
            f"row sequence must have {len(COLUMNS)} values in order {COLUMNS}, got {len(seq)}"
        )
    track, start_sec, duration_sec, species, probability = seq
    return ResultRow(
        track=str(track),
        start_sec=float(start_sec),
        duration_sec=float(duration_sec),
        species=str(species),
        probability=float(probability),
    )


class CsvWriter:
    """Write classification results to ``results.csv`` (+ optional sidecar).

    Two usage modes:

    * one-shot: ``CsvWriter().write(rows, path, meta=...)``
    * streaming: ``with CsvWriter.open(path) as w: w.write_row(row)`` and then
      ``w.write_meta(meta)`` (or pass ``meta`` to ``open``).

    The header and column order are fixed by ``COLUMNS``.
    """

    def __init__(self, *, dialect: str = "excel") -> None:
        self._dialect = dialect

    # ----------------------------------------------------------------- one-shot
    def write(
        self,
        rows: Iterable[ResultRow | dict[str, object] | Sequence[object]],
        path: str | Path,
        *,
        meta: TrapMeta | None = None,
    ) -> Path:
        """Write all ``rows`` to ``path`` and (if given) the sidecar.

        Returns the resolved CSV path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = self._make_writer(fh)
            writer.writeheader()
            for row in rows:
                writer.writerow(_coerce_row(row).as_csv_dict())
        if meta is not None:
            self.write_meta(meta, path)
        return path

    # ---------------------------------------------------------------- streaming
    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        meta: TrapMeta | None = None,
        dialect: str = "excel",
    ) -> "_StreamingCsvWriter":
        """Open a streaming writer as a context manager."""
        return _StreamingCsvWriter(cls(dialect=dialect), Path(path), meta)

    # --------------------------------------------------------------------- meta
    @staticmethod
    def meta_path(csv_path: str | Path) -> Path:
        """Sidecar path next to ``results.csv`` -> ``results_meta.json``."""
        csv_path = Path(csv_path)
        return csv_path.with_name(f"{csv_path.stem}_meta.json")

    def write_meta(self, meta: TrapMeta, csv_path: str | Path) -> Path:
        """Write the sidecar JSON next to ``csv_path``. Returns sidecar path."""
        sidecar = self.meta_path(csv_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(meta.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return sidecar

    # ------------------------------------------------------------------ private
    def _make_writer(self, fh: IO[str]) -> "csv.DictWriter[str]":
        return csv.DictWriter(
            fh,
            fieldnames=list(COLUMNS),
            dialect=self._dialect,
            extrasaction="raise",
        )


class _StreamingCsvWriter:
    """Context-managed incremental writer; header written on enter."""

    def __init__(self, owner: CsvWriter, path: Path, meta: TrapMeta | None) -> None:
        self._owner = owner
        self._path = path
        self._meta = meta
        self._fh: IO[str] | None = None
        self._writer: "csv.DictWriter[str] | None" = None

    def __enter__(self) -> "_StreamingCsvWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="", encoding="utf-8")
        self._writer = self._owner._make_writer(self._fh)
        self._writer.writeheader()
        return self

    def write_row(self, row: ResultRow | dict[str, object] | Sequence[object]) -> None:
        if self._writer is None:
            raise RuntimeError("writer used outside of context manager")
        self._writer.writerow(_coerce_row(row).as_csv_dict())

    def write_rows(
        self, rows: Iterable[ResultRow | dict[str, object] | Sequence[object]]
    ) -> None:
        for row in rows:
            self.write_row(row)

    def write_meta(self, meta: TrapMeta) -> Path:
        return self._owner.write_meta(meta, self._path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        # Only emit the sidecar on a clean exit so a crashed run doesn't claim
        # provenance over a truncated CSV.
        if exc_type is None and self._meta is not None:
            self._owner.write_meta(self._meta, self._path)
