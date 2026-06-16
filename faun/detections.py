"""Detections: the central detection abstraction shared across the pipeline.

A :class:`Detection` ties an extracted audio segment (on the original recording
timeline) to its trap, source recording, on-disk clip, and an ordered list of
:class:`Label`\\ s. Labels carry provenance (``source``) and a lifecycle
``status`` so model pseudo-labels and human ground truth coexist in one record.

Persistence is line-delimited JSON (JSONL), one detection per line, written
atomically (tmp + rename) so a Phase-2 writer and the CLI export can both rely
on a stable on-disk contract.

stdlib + typing only — reuses ``Segment`` / ``Prediction`` from sibling
packages; no heavy ML imports here.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from faun.classification import Prediction
from faun.segmentation import Segment

__all__ = [
    "SOURCE_PERCH",
    "SOURCE_BIRDNET",
    "SOURCE_YAMNET_PROBE",
    "SOURCE_STUB",
    "SOURCE_EXPERT",
    "SOURCE_RANGER",
    "STATUS_PSEUDO",
    "STATUS_CONFIRMED",
    "STATUS_CORRECTED",
    "Label",
    "Detection",
    "is_ground_truth",
    "write_detections",
    "read_detections",
]

# --- Label provenance (source) -------------------------------------------------
SOURCE_PERCH = "model:perch"
SOURCE_BIRDNET = "model:birdnet"
SOURCE_YAMNET_PROBE = "model:yamnet-probe"
SOURCE_STUB = "model:stub"
SOURCE_EXPERT = "expert:ornithologist"
SOURCE_RANGER = "operator:ranger"

# --- Label lifecycle (status) --------------------------------------------------
STATUS_PSEUDO = "pseudo"
STATUS_CONFIRMED = "confirmed"
STATUS_CORRECTED = "corrected"

#: Status values that, combined with a human source, mark a label ground truth.
_GROUND_TRUTH_STATUSES = frozenset({STATUS_CONFIRMED, STATUS_CORRECTED})

#: Source prefixes that denote a human-provided label.
_HUMAN_PREFIXES = ("expert:", "operator:")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Label:
    """A single (species, provenance, lifecycle) annotation for a detection."""

    species: str
    probability: float | None
    source: str
    status: str
    ts: str

    @classmethod
    def now(
        cls,
        species: str,
        probability: float | None,
        source: str,
        status: str,
    ) -> "Label":
        """Create a label stamped with the current UTC time (ISO-8601)."""
        return cls(
            species=species,
            probability=probability,
            source=source,
            status=status,
            ts=_utcnow(),
        )

    @classmethod
    def from_prediction(
        cls,
        pred: Prediction,
        source: str,
        status: str = STATUS_PSEUDO,
    ) -> "Label":
        """Lift a classifier :class:`Prediction` into a (pseudo) label."""
        return cls.now(
            species=pred.species,
            probability=pred.probability,
            source=source,
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "probability": self.probability,
            "source": self.source,
            "status": self.status,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Label":
        return cls(
            species=data["species"],
            probability=data["probability"],
            source=data["source"],
            status=data["status"],
            ts=data["ts"],
        )


@dataclass
class Detection:
    """An extracted segment with its provenance and ordered labels."""

    detection_id: str
    trap_id: str
    source_file: str
    segment: Segment
    segment_path: str
    labels: list[Label] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        trap_id: str,
        source_file: str,
        segment: Segment,
        labels: list[Label] | None = None,
        detection_id: str | None = None,
    ) -> "Detection":
        """Create a detection with a fresh uuid4-hex id and derived clip path."""
        det_id = detection_id or uuid.uuid4().hex
        return cls(
            detection_id=det_id,
            trap_id=trap_id,
            source_file=source_file,
            segment=segment,
            segment_path=f"segments/{det_id}.wav",
            labels=list(labels) if labels else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "trap_id": self.trap_id,
            "source_file": self.source_file,
            "segment": {
                "start_s": self.segment.start_s,
                "duration_s": self.segment.duration_s,
            },
            "segment_path": self.segment_path,
            "labels": [label.to_dict() for label in self.labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        seg = data["segment"]
        return cls(
            detection_id=data["detection_id"],
            trap_id=data["trap_id"],
            source_file=data["source_file"],
            segment=Segment(
                start_s=seg["start_s"],
                duration_s=seg["duration_s"],
            ),
            segment_path=data["segment_path"],
            labels=[Label.from_dict(d) for d in data["labels"]],
        )


def is_ground_truth(label: Label) -> bool:
    """True iff ``label`` is human-provided AND confirmed/corrected.

    Model sources ("model:*", always status "pseudo") are never ground truth.
    """
    return (
        label.source.startswith(_HUMAN_PREFIXES)
        and label.status in _GROUND_TRUTH_STATUSES
    )


def write_detections(
    path: str | os.PathLike[str], detections: Iterable[Detection]
) -> None:
    """Atomically write detections as JSONL (one object per line).

    Writes to ``.<name>.tmp`` in the same directory then ``os.replace``\\ s it
    into place, mirroring ``faun.jobs`` manifest persistence.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    lines = [json.dumps(det.to_dict(), ensure_ascii=False) for det in detections]
    payload = "".join(f"{line}\n" for line in lines)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, path)


def read_detections(path: str | os.PathLike[str]) -> list[Detection]:
    """Read a JSONL detections file, tolerant of trailing/blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [
        Detection.from_dict(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]
