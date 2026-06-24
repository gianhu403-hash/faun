"""Shared infrastructure for the frozen golden-baseline gate (SF-B2 / Verif#3).

The honesty gate for every flagged wave (B/C/D/E) is: *with all new flags OFF,
the pipeline output is byte-for-byte identical to a reference frozen from
``main`` at ``8c24ca9`` (the merge-base before the Waves 2–4 session).*

A regen-from-current-code comparison would only prove "the flag is a no-op",
not "the new code matches old main" — so the reference is **frozen on disk**
(``tests/fixtures/golden/``) and this module is the SINGLE place that produces
the live output to compare against it, so the generator script and the test
agree by construction.

Determinism notes:
- ``results.csv`` columns (``track,start_sec,duration_sec,species,probability``)
  carry no id/timestamp and round to fixed precision → byte-reproducible across
  processes and commits. Frozen and compared verbatim.
- ``detections.jsonl`` lines carry a random ``detection_id`` (uuid4), a derived
  ``segment_path``, a wall-clock ``Label.ts``, and (post-FR-006) a
  ``prob_calibrated`` sidecar field that is ``null`` in the OFF state. Those four
  are intrinsically per-run / OFF-state and carry no scored-output signal, so the
  jsonl is compared in a NORMALIZED form with exactly those keys stripped — the
  species / probability / segment-bounds / source / status / row-order that the
  gate actually guards remain fully compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from faun.classification import Prediction

#: Frozen reference directory (committed).
GOLDEN_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "results_main_8c24ca9.csv"
GOLDEN_JSONL = GOLDEN_DIR / "detections_main_8c24ca9.jsonl"

SR = 48_000

#: jsonl keys stripped before comparison — per-run identity / OFF-state sidecar,
#: not scored output. See module docstring.
_DETECTION_DROP = ("detection_id", "segment_path")
_LABEL_DROP = ("ts", "prob_calibrated")


class GoldenClf:
    """Deterministic, real-logit-shaped fake classifier (NOT Stub, NOT TF).

    Returns a fixed top-5 of scientific names with raw-logit-valued
    "probabilities" (Perch 2 reports logits > 1 on raw180, e.g. 13.3) so the
    golden exercises the multi-label, logit-valued serialization path rather than
    Stub's single ``Turdus merula`` row. ``vocab`` mirrors a real adapter's label
    vocabulary so the same fake can be reused where a ``vocab_provider`` is
    needed; the golden run leaves masking OFF, so it is unused there.
    """

    #: Full label vocabulary (superset of the emitted predictions).
    vocab = [
        "Turdus merula",
        "Erithacus rubecula",
        "Fringilla coelebs",
        "Cyanistes caeruleus",
        "Sturnus vulgaris",
        "Parus major",
        "Phylloscopus collybita",
        "Periparus ater",
    ]

    _PREDICTIONS = [
        ("Turdus merula", 13.2734),
        ("Erithacus rubecula", 11.8421),
        ("Fringilla coelebs", 9.5102),
        ("Cyanistes caeruleus", 7.4410),
        ("Sturnus vulgaris", 5.1290),
    ]

    def classify(self, segment, sr):  # noqa: ANN001 — protocol signature
        return [Prediction(name, logit) for name, logit in self._PREDICTIONS]


def make_trap_dir(root: Path, trap_id: str = "A1") -> Path:
    """Synthetic trap folder, identical to ``test_e2e._make_trap_dir``.

    Seeded RNG + a fixed 3 kHz burst at 2.5–3.0 s → deterministic segmentation.
    """
    trap = root / trap_id
    trap.mkdir(parents=True)
    t = np.linspace(0, 6, 6 * SR, endpoint=False)
    sig = 0.005 * np.random.default_rng(7).standard_normal((6 * SR, 2))
    burst = np.sin(2 * np.pi * 3000 * t) * ((t > 2.5) & (t < 3.0))
    sig[:, 0] += burst
    sig[:, 1] += burst
    sf.write(trap / "REC_20260610_213000.wav", sig, SR)
    (trap / "info.txt").write_text(
        "date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel\n"
        "2026-06-10,21:30:00,32.95,60.55,3.9,14.2,71,"
        "REC_20260610_213000.wav,48000,auto,stereo\n",
        encoding="utf-8",
    )
    return trap


def normalize_jsonl(raw_text: str) -> str:
    """Strip per-run / OFF-state keys, re-dump one stable object per line."""
    out_lines = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for key in _DETECTION_DROP:
            obj.pop(key, None)
        for label in obj.get("labels", []):
            for key in _LABEL_DROP:
                label.pop(key, None)
        out_lines.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def run_golden(tmp_path: Path) -> tuple[bytes, str]:
    """Run ``run_pipeline`` with all new flags OFF; return (csv_bytes, norm_jsonl).

    Explicitly neutralizes every flag this session adds so the call reproduces
    the pre-session default path regardless of ambient env.
    """
    from faun.api import run_pipeline
    from faun.settings import get_settings

    # Every flag this session introduces, pinned to its OFF/default value.
    import os

    for var in (
        "FAUN_SPECIES_ALLOWLIST",
        "FAUN_PROB_SMOOTHING",
        "FAUN_PRESENCE_GATE_K",
        "PERCH_V2_CALIBRATOR_PATH",
    ):
        os.environ.pop(var, None)
    get_settings.cache_clear()

    data = tmp_path / "data"
    make_trap_dir(data)
    job_dir = tmp_path / "job"
    csv_path = run_pipeline(job_dir, str(data), classifier=GoldenClf())
    csv_bytes = csv_path.read_bytes()
    jsonl_raw = (job_dir / "detections.jsonl").read_text(encoding="utf-8")
    return csv_bytes, normalize_jsonl(jsonl_raw)
