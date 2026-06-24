# Changelog

All notable changes to the Faun v2 pipeline are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); architecture decisions
live alongside in `docs/adr/`.

This file starts at the production-ready vector (Wave 1). Earlier history is in
the git log and `~/.claude` project memory.

## [Unreleased]

### Wave 4 — prototypical probe with a negative (background) class (ADR-0008)

A new probe trainer beside `train_probe_cv` that gives the **probe** path an
explicit "none of the birds" outcome on out-of-distribution (non-bird / noise)
audio. The existing LogReg trainer, the served `PerchProbeAdapter`, and the
frozen contract are all **unchanged** — this is purely additive.

#### Added
- **Prototypical probe (FR-007).** `faun.retraining.train_prototype_probe(X, y,
  *, negatives=None, metric="cosine", temperature=1.0)` returns a small picklable
  `_PrototypeProbe`: one centroid per class, `predict_proba` = softmax over
  per-class similarity logits (rows sum to 1, in `[0, 1]`). Deterministic (no
  seed, no torch), dimension-agnostic, and a drop-in for the **unchanged**
  `PerchProbeAdapter` (same `predict_proba` / `classes_` surface; pickles via
  `save_probe` / `load_probe` and loads through `YAMNetAdapter._load_probe`).
- **Negative (background) class.** `NEGATIVE_CLASS = "__negative__"` — a distinct
  *class* the probe predicts (not the `unknown` token, not `STATUS_REJECTED`).
  When `negatives` (non-bird embeddings) is passed, a dedicated negative prototype
  is added, so an OOD embedding near it is classified negative instead of a
  confident bird. `negatives=None` → a plain multiclass prototypical probe.
- **Head-to-head eval (opt-in, cluster-only).** `scripts/eval_inatsounds_perchv2.py`
  gains `--prototype` (and `--negatives-from <dir>`): trains the prototype probe
  on the **same disjoint train split** and reports its held-out macro-F1 next to
  the LogReg-0.834 baseline on the **same disjoint val** (apples-to-apples; the
  `#H2` leakage guards are unchanged). Writes `prototype_heldout_macro_f1` into
  the JSON summary. Without the flag the script's default output is unchanged.

#### Unchanged (guarantees)
- `train_probe_cv`, `_ConstantProbe`, `save_probe` / `load_probe`, and the served
  `PerchProbeAdapter` are untouched; no `faun/INTERFACES.md` signature changes.
- No module-level `import tensorflow` / `import torch` — the probe is pure numpy.

#### Notes
- The **mechanism** ships here; whether the prototype probe **beats** the 0.834
  LogReg baseline (SC-D2) is a cluster measurement that is **not** run in this
  wave (it needs Perch 2 embeddings of iNatSounds + non-bird clips on the
  cluster). The probe is not wired into production; `train_probe_cv` stays the
  default trainer.
- The 0.834 baseline is **iNatSounds held-out macro-F1, not raw180 serving
  accuracy** — iNat-head-vs-onset domain shift; see
  `experiments/report/METRICS_HONESTY.md` §10.3–10.4. raw180 species accuracy
  remains unmeasured.

### Wave 3 — presence soft-gate + serve-time calibration (ADR-0007)

Two additive signals inside `Perch2Adapter.classify`, both computed from the
SAME logits (zero extra inference) and **OFF by default** — with
`FAUN_PRESENCE_GATE_K=0` (default) and no `PERCH_V2_CALIBRATOR_PATH`, the served
`probability` is the raw logit, byte-for-byte (golden gate green).

#### Added
- **Presence soft-gate (FR-003).** `_load_bird_mask()` reads
  `assets/perch_v2_ebird_classes.csv` (bird = eBird code ≠ `no_ebird_code`) to
  weight predictions by the segment's bird-mass. `k=FAUN_PRESENCE_GATE_K`: `k=0`
  is a LITERAL raw-logit no-op (not the formula at k=0, which would silently swap
  the logit for a softmax prob); `k>0` returns `clamp01(p_species·(1+p_bird·k))`.
  Math in unit-tested free functions `_softmax` / `bird_presence_mass` /
  `apply_presence_gate`. Fail-open: missing/unreadable/length-mismatched asset
  disables the gate with a warning.
- **Serve-time calibration (FR-006-serve).** `Prediction.prob_calibrated:
  float|None` (optional, default None). When `PERCH_V2_CALIBRATOR_PATH` points at
  a pickled `TemperatureCalibrator`, `classify` fills `prob_calibrated` with
  `softmax(logits/T)[i] ∈ [0,1]` from the RAW logits, independent of the gate.
  `Label.from_prediction` threads it into `detections.jsonl` (no `api.py`
  change). Raw `probability` never overwritten; never a CSV column.
- `presence_gate_k` (`FAUN_PRESENCE_GATE_K`, parsed `positive=False` so 0 is a
  valid OFF) and `perch_v2_calibrator_path` (`PERCH_V2_CALIBRATOR_PATH`) in
  `faun.settings`.

#### Unchanged (guarantees)
- Frozen CSV columns; `Prediction` extended only by an optional field (every
  `Prediction(species, probability)` call stays valid); Perch 2 backbone
  (`_prepare`/peak-norm) untouched; raw `probability` never rewritten.
- Default-path output (k=0, no calibrator) is byte-for-byte the `main@8c24ca9`
  golden baseline.

#### Notes
- The mechanism ships, not a fitted `k` or a deployed calibrator. Calibration is
  only measurable on iNatSounds heads (domain shift, METRICS_HONESTY §10.4);
  tuning `k` wants reserve audio. Cluster validation is a best-effort follow-up,
  not a merge gate.

### Wave 2 — honest segmentation + per-recording smoothing (ADR-0006)

Recall and denoising levers for bird song (the onset detector was tuned for
transients). Additive and **OFF by default** — with all flags unset the produced
segments and `results.csv`/`detections.jsonl` are byte-for-byte the frozen
`main@8c24ca9` baseline (golden gate). All segment pruning happens INSIDE
`SegmentExtractor.extract` before it returns, so the clip↔detection row-alignment
(ADR-0003) is preserved.

#### Added
- **Dense windows (FR-002).** `SegmentExtractor` keyword knobs `dense_windows`
  (a `window_s`-long grid hopped by `hop_s`, default 5 s / 2.5 s) for sustained
  song the transient onset detector misses. The onset path stays the default.
- **Silence pre-filter (FR-002b).** `silence_filter` drops produced windows whose
  whole-window 16 kHz RMS is at/below `faun.ml.onset.MIN_ABSOLUTE_ENERGY`.
- **Temporal-IoU NMS (FR-002c).** `nms_iou` generalizes the legacy overlap drop to
  a greedy temporal intersection-over-union suppression (`None` = legacy
  behaviour, exactly). Temporal only — `extract` is classifier-free.
- **Per-recording probability smoothing (FR-004).** New sidecar
  `prob_smoothed.json` (`faun.output.prob_smoothed` / `write_prob_smoothed`):
  per `(recording, species)`, a time-ordered centered moving average of detection
  probabilities. Written by `run_pipeline` **only when `FAUN_PROB_SMOOTHING` is
  set**; raw `probability` and the CSV are untouched.
- `prob_smoothing` setting in `faun.settings` (`FAUN_PROB_SMOOTHING`, default off).

#### Unchanged (guarantees)
- Frozen `extract(waveform, sr)` signature, `results.csv` columns, and the
  `OnsetDetector` public API (`tests/test_onset.py` untouched).
- Default-path output (all knobs off, `FAUN_PROB_SMOOTHING` unset) is byte-for-byte
  the `main@8c24ca9` golden baseline.

#### Deferred (documented in ADR-0006)
- An operator seam (env/settings) for `dense_windows`/`silence_filter`/`nms_iou`:
  this wave lands them as `SegmentExtractor` constructor primitives with
  direct-construction tests; `run_pipeline` still builds a default extractor, so
  the served path is unchanged. Wiring them through `faun.settings` (the seam
  `prob_smoothing` already has) is a low-risk follow-up.

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
