# Changelog

All notable changes to the Faun v2 pipeline are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); architecture decisions
live alongside in `docs/adr/`.

This file starts at the production-ready vector (Wave 1). Earlier history is in
the git log and `~/.claude` project memory.

## [Unreleased]

### Probability calibration — temperature scaling + reject status (ADR-0005)

Fixes the misleading "logits as probability" on the served zero-shot path:
`Perch2Adapter` reports raw logits (> 1 on raw180) in the `probability` field.
Additive and OFF by default — the raw `probability` and the frozen CSV are
untouched; a calibrated value travels only in the `detections.jsonl` sidecar.

#### Added
- **Temperature calibration** in `faun.retraining` (Guo et al. 2017):
  `TemperatureCalibrator`, `fit_temperature` (NLL-minimised scalar `T`, `T=1`
  fallback), `apply_calibration` (identity pass-through when no calibrator), and
  `expected_calibration_error` (ECE). Pure numpy/scipy — no new dependency.
- **`STATUS_REJECTED = "rejected"`** in `faun.detections` — model abstain /
  negative-class lifecycle status, never ground truth, distinct from the
  `unknown` species token.
- **`Label.prob_calibrated`** optional field (`detections.jsonl` sidecar only,
  never a CSV column); back-compatible (`None` for pre-FR-006 records).

#### Changed
- `python-multipart` `0.0.22 → 0.0.31` — clears 5 pip-audit CVEs (the dep is a
  transitive FastAPI form-parser, unused by any route; pure hygiene).

#### Deferred (documented in ADR-0005)
- Serve-time population of `prob_calibrated` (needs the full per-class logit
  vector, not exposed by the top-k `classify` contract).
- raw180 transfer validation (needs ornithologist labels; calibration is only
  measurable on iNatSounds clip heads — domain shift, METRICS_HONESTY §10.4).

### Wave 1 — regional allow-list + presence aggregate + demo (ADR-0004)

Quality levers that need neither GPU nor a human-labelled corpus. Every change
is additive and **OFF by default** — with all flags unset, `results.csv` is
byte-for-byte identical to the previous build (golden-CSV diff gate).

#### Added
- **Regional species allow-list (`MaskedClassifier`, FR-001).** Wraps the
  `SpeciesClassifier` protocol in `faun.api._build_classifier` (works for Perch 2
  and any future probe) to restrict the served output to listed species.
  Case-insensitive, underscore-tolerant name matching (`_species_key`). Enabled
  by `FAUN_SPECIES_ALLOWLIST` (path, or `default`/`reserve` for the bundled
  seed); unset → no masking. Each dropped prediction logs `masked_out=<species>`.
- **Default reserve checklist** `faun/data/reserve_checklist.txt` — 69 Palearctic
  forest binomials (mirrors `scripts/extract_inatsounds_subset.py:RESERVE`),
  refinable by the ornithologist.
- **Fail-loud, fail-open coverage gate.** If too few allow-list names match the
  classifier's own label vocabulary (typo / wrong checklist / `species_<i>`
  fallback), the mask disables itself with a warning instead of emptying the CSV.
- **Per-trap, per-day species presence aggregate (FR-005).** New sidecar
  `species_presence.json` (`faun.output.species_presence` /
  `write_species_presence`), written by `run_pipeline` next to
  `detections.jsonl`: each `(trap, day)` group lists species, detection count,
  and max/mean probability. Day is derived from the filename via `faun.ingest`.
- `FAUN_SPECIES_ALLOWLIST` setting in `faun.settings`.

#### Changed
- `faun.api._build_classifier` now applies the allow-list when configured;
  `_classifier_source` unwraps `MaskedClassifier.inner` so detection provenance
  still records the real backbone (e.g. `model:perch-v2`).

#### Unchanged (guarantees)
- Frozen `results.csv` columns and `faun/INTERFACES.md` signatures.
- Perch 2 backbone (`_prepare` / peak-norm) and the production zero-shot output
  while `FAUN_SPECIES_ALLOWLIST` is unset.

#### Notes
- The 0.834 macro-F1 (iNatSounds, 50 reserve species) is **not** raw180 serving
  accuracy — see `experiments/report/METRICS_HONESTY.md` §10.4. raw180 accuracy
  remains unmeasured (needs ornithologist labelling).
