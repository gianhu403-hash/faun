"""Tests for faun.output — CsvWriter + results_meta.json sidecar.

Covers the frozen column order/header, rounding, row coercion (dataclass /
dict / tuple), the sidecar metadata, and streaming writes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from faun.output import (
    COLUMNS,
    PIPELINE_VERSION,
    CsvWriter,
    ResultRow,
    TrapMeta,
    species_presence,
    write_species_presence,
)
from faun.detections import (
    Detection,
    Label,
    SOURCE_PERCH_V2,
    SOURCE_RANGER,
    STATUS_CORRECTED,
    STATUS_PSEUDO,
)
from faun.segmentation import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return reader.fieldnames or [], rows


# ---------------------------------------------------------------------------
# Column order / header
# ---------------------------------------------------------------------------


class TestColumns:
    def test_frozen_column_order(self) -> None:
        assert COLUMNS == (
            "track",
            "start_sec",
            "duration_sec",
            "species",
            "probability",
        )

    def test_header_matches_columns(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([], out)
        header, rows = _read_csv(out)
        assert header == list(COLUMNS)
        assert rows == []

    def test_raw_first_line_is_exact_header(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([], out)
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "track,start_sec,duration_sec,species,probability"


# ---------------------------------------------------------------------------
# Row writing + rounding
# ---------------------------------------------------------------------------


class TestRowWriting:
    def test_dataclass_rows_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        rows = [
            ResultRow("A1_0001.wav", 12.0, 5.0, "Turdus merula", 0.91),
            ResultRow("A1_0001.wav", 30.5, 2.25, "unknown", 0.42),
        ]
        CsvWriter().write(rows, out)
        _, read = _read_csv(out)
        assert len(read) == 2
        assert read[0]["track"] == "A1_0001.wav"
        assert read[0]["species"] == "Turdus merula"

    def test_probability_rounded_to_4(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([ResultRow("t.wav", 0.0, 1.0, "x", 0.123456789)], out)
        _, read = _read_csv(out)
        assert read[0]["probability"] == "0.1235"

    def test_seconds_rounded_to_2(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([ResultRow("t.wav", 1.23456, 9.87654, "x", 0.5)], out)
        _, read = _read_csv(out)
        assert read[0]["start_sec"] == "1.23"
        assert read[0]["duration_sec"] == "9.88"

    def test_dict_rows_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write(
            [
                {
                    "track": "t.wav",
                    "start_sec": 1.0,
                    "duration_sec": 2.0,
                    "species": "y",
                    "probability": 0.5,
                }
            ],
            out,
        )
        _, read = _read_csv(out)
        assert read[0]["species"] == "y"

    def test_tuple_rows_in_column_order(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([("t.wav", 1.0, 2.0, "z", 0.5)], out)
        _, read = _read_csv(out)
        assert read[0]["track"] == "t.wav"
        assert read[0]["species"] == "z"

    def test_wrong_length_tuple_rejected(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        with pytest.raises(ValueError):
            CsvWriter().write([("t.wav", 1.0, 2.0)], out)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "results.csv"
        CsvWriter().write([ResultRow("t.wav", 0.0, 1.0, "x", 0.5)], out)
        assert out.is_file()


# ---------------------------------------------------------------------------
# Sidecar metadata
# ---------------------------------------------------------------------------


class TestSidecar:
    def test_sidecar_written_next_to_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        meta = TrapMeta(
            trap_id="A1",
            lat=59.93,
            lon=30.31,
            files=["A1_0001.wav", "A1_0002.wav"],
        )
        CsvWriter().write(
            [ResultRow("A1_0001.wav", 0.0, 1.0, "x", 0.5)], out, meta=meta
        )
        sidecar = tmp_path / "results_meta.json"
        assert sidecar.is_file()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["trap_id"] == "A1"
        assert data["lat"] == 59.93
        assert data["lon"] == 30.31
        assert data["files"] == ["A1_0001.wav", "A1_0002.wav"]
        assert data["pipeline_version"] == PIPELINE_VERSION

    def test_meta_path_derivation(self) -> None:
        assert CsvWriter.meta_path("/a/b/results.csv").name == "results_meta.json"
        assert CsvWriter.meta_path("/a/b/out.csv").name == "out_meta.json"

    def test_no_sidecar_when_meta_absent(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([], out)
        assert not (tmp_path / "results_meta.json").exists()

    def test_extra_flattened_but_does_not_shadow(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        meta = TrapMeta(
            trap_id="A2",
            extra={"gain": "AGC", "trap_id": "SHOULD_NOT_WIN", "battery": 3.7},
        )
        CsvWriter().write([], out, meta=meta)
        data = json.loads((tmp_path / "results_meta.json").read_text("utf-8"))
        assert data["trap_id"] == "A2"  # canonical key not shadowed
        assert data["gain"] == "AGC"
        assert data["battery"] == 3.7

    def test_default_coords_none(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        CsvWriter().write([], out, meta=TrapMeta(trap_id="A3"))
        data = json.loads((tmp_path / "results_meta.json").read_text("utf-8"))
        assert data["lat"] is None
        assert data["lon"] is None
        assert data["files"] == []


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_streaming_rows_and_meta(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        meta = TrapMeta(trap_id="A4", files=["A4_0001.wav"])
        with CsvWriter.open(out, meta=meta) as w:
            w.write_row(ResultRow("A4_0001.wav", 0.0, 1.0, "a", 0.5))
            w.write_rows(
                [
                    ResultRow("A4_0001.wav", 1.0, 1.0, "b", 0.6),
                    {
                        "track": "A4_0001.wav",
                        "start_sec": 2.0,
                        "duration_sec": 1.0,
                        "species": "c",
                        "probability": 0.7,
                    },
                ]
            )
        header, read = _read_csv(out)
        assert header == list(COLUMNS)
        assert [r["species"] for r in read] == ["a", "b", "c"]
        sidecar = tmp_path / "results_meta.json"
        assert json.loads(sidecar.read_text("utf-8"))["trap_id"] == "A4"

    def test_streaming_no_sidecar_on_exception(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        meta = TrapMeta(trap_id="A5")
        with pytest.raises(RuntimeError):
            with CsvWriter.open(out, meta=meta) as w:
                w.write_row(ResultRow("t.wav", 0.0, 1.0, "x", 0.5))
                raise RuntimeError("boom")
        # CSV header was written, but sidecar must NOT claim provenance.
        assert out.is_file()
        assert not (tmp_path / "results_meta.json").exists()

    def test_write_row_outside_context_raises(self, tmp_path: Path) -> None:
        w = CsvWriter.open(tmp_path / "results.csv")
        with pytest.raises(RuntimeError):
            w.write_row(ResultRow("t.wav", 0.0, 1.0, "x", 0.5))


# ---------------------------------------------------------------------------
# Species presence aggregate (species_presence.json) — FR-005, ADR-0004
# ---------------------------------------------------------------------------


def _det(trap: str, fname: str, *labels: Label) -> Detection:
    return Detection.new(
        trap_id=trap,
        source_file=fname,
        segment=Segment(1.0, 5.0),
        labels=list(labels),
    )


def _pseudo(species: str, prob: float) -> Label:
    return Label.now(species, prob, SOURCE_PERCH_V2, STATUS_PSEUDO)


class TestSpeciesPresence:
    def test_groups_by_trap_and_day(self) -> None:
        dets = [
            _det("A1", "REC_20260610_213000.wav", _pseudo("Turdus merula", 0.91)),
            _det("A1", "REC_20260610_220000.wav", _pseudo("Turdus merula", 0.80)),
            _det("A1", "REC_20260611_060000.wav", _pseudo("Parus major", 0.70)),
        ]
        out = species_presence(dets)
        keys = [(g["trap_id"], g["day"]) for g in out["groups"]]
        assert keys == [("A1", "2026-06-10"), ("A1", "2026-06-11")]
        day0 = out["groups"][0]
        assert day0["n_detections"] == 2
        assert day0["species"][0]["species"] == "Turdus merula"
        assert day0["species"][0]["detections"] == 2
        assert day0["species"][0]["max_probability"] == 0.91
        assert day0["species"][0]["mean_probability"] == 0.855

    def test_ground_truth_label_wins(self) -> None:
        det = _det(
            "A2",
            "REC_20260610_050000.wav",
            _pseudo("Strix aluco", 0.30),
            Label.now("Bubo bubo", None, SOURCE_RANGER, STATUS_CORRECTED),
        )
        out = species_presence([det])
        sp = out["groups"][0]["species"]
        assert [s["species"] for s in sp] == ["Bubo bubo"]
        # human label carries no probability -> stats are null, count still 1.
        assert sp[0]["detections"] == 1
        assert sp[0]["max_probability"] is None
        assert sp[0]["mean_probability"] is None

    def test_unparseable_filename_day_is_null(self) -> None:
        out = species_presence([_det("A3", "weird.wav", _pseudo("Corvus corax", 0.5))])
        assert out["groups"][0]["day"] is None

    def test_count_invariant_total_equals_detections(self) -> None:
        """Sum of group n_detections == number of detections; per-species sum ==
        attributed (an all-masked-out detection is counted but unattributed)."""
        dets = [
            _det("A1", "REC_20260610_213000.wav", _pseudo("Turdus merula", 0.9)),
            _det("A1", "REC_20260610_220000.wav", _pseudo("Parus major", 0.6)),
            _det("A1", "REC_20260610_223000.wav"),  # no labels (all masked out)
        ]
        out = species_presence(dets)
        total = sum(g["n_detections"] for g in out["groups"])
        attributed = sum(s["detections"] for g in out["groups"] for s in g["species"])
        assert total == len(dets) == 3
        assert attributed == 2  # the unlabeled detection is counted, not attributed

    def test_best_label_is_argmax(self) -> None:
        det = _det(
            "A1",
            "REC_20260610_213000.wav",
            _pseudo("unknown", 0.42),
            _pseudo("Turdus merula", 0.91),
        )
        out = species_presence([det])
        assert [s["species"] for s in out["groups"][0]["species"]] == ["Turdus merula"]

    def test_species_sorted_by_count_then_name(self) -> None:
        dets = [
            _det("A1", "REC_20260610_010000.wav", _pseudo("Parus major", 0.9)),
            _det("A1", "REC_20260610_020000.wav", _pseudo("Turdus merula", 0.9)),
            _det("A1", "REC_20260610_030000.wav", _pseudo("Turdus merula", 0.9)),
        ]
        sp = species_presence(dets)["groups"][0]["species"]
        assert [s["species"] for s in sp] == ["Turdus merula", "Parus major"]

    def test_empty_detections(self) -> None:
        out = species_presence([])
        assert out["groups"] == []
        assert out["pipeline_version"] == PIPELINE_VERSION

    def test_write_species_presence_round_trips(self, tmp_path: Path) -> None:
        det = _det("A1", "REC_20260610_213000.wav", _pseudo("Turdus merula", 0.91))
        out_path = write_species_presence(tmp_path / "species_presence.json", [det])
        assert out_path.is_file()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["groups"][0]["trap_id"] == "A1"
        assert data["groups"][0]["species"][0]["species"] == "Turdus merula"
