"""Frozen golden-baseline gate (SF-B2 / Verif#3).

These two tests are the honesty gate every flagged wave (B/C/D/E) must pass
before merge: with all new flags OFF, the pipeline output must equal a reference
**frozen on disk** from ``main`` at ``8c24ca9`` — not regenerated from the code
under test. See ``golden_util`` for the determinism rationale and the exact
normalization applied to ``detections.jsonl``.

If a wave legitimately and intentionally changes the default-path output, that is
a contract change: it must be called out, the frozen reference regenerated, and
the ADR/CHANGELOG updated — never silently.
"""

from __future__ import annotations

from pathlib import Path

from golden_util import GOLDEN_CSV, GOLDEN_JSONL, run_golden


def test_golden_csv_byte_identical(tmp_path: Path) -> None:
    csv_bytes, _ = run_golden(tmp_path)
    assert csv_bytes == GOLDEN_CSV.read_bytes(), (
        "results.csv drifted from the frozen main@8c24ca9 baseline with all flags "
        "OFF — a default-path output change. If intentional, regenerate "
        "tests/fixtures/golden/ and update the ADR/CHANGELOG; otherwise it is a "
        "regression."
    )


def test_golden_detections_normalized_identical(tmp_path: Path) -> None:
    _, norm_jsonl = run_golden(tmp_path)
    assert norm_jsonl == GOLDEN_JSONL.read_text(encoding="utf-8"), (
        "detections.jsonl (normalized: detection_id/segment_path/ts/prob_calibrated "
        "stripped) drifted from the frozen main@8c24ca9 baseline with all flags OFF."
    )
