# ADR 0007 — Presence soft-gate + serve-time probability calibration

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/wave3-presence-calibration`

## Context

Two weaknesses in the served zero-shot `Perch2Adapter` output, both addressable
from the logits the model already produces — no extra inference, no GPU, no
labels:

1. **No use of the model's own non-bird knowledge.** Perch 2's 14795-class head
   is **9706 birds** (rows with an eBird code) + **5089 non-bird** classes
   (`no_ebird_code`, the FSD50K noise rows: Wind, Vehicle, Speech, …). On noisy
   trap audio a non-bird segment can still surface a confident *bird* species.
   The model already tells us how much of the segment is bird vs noise — the
   softmax mass over the bird columns — but nothing uses it. (FR-003: the spike
   verdict `perch2-native-logits` — bird-mass from the same logits — over a
   separate CLAP presence model, which was rejected.)

2. **`probability` carries a raw logit.** `Perch2Adapter.classify` returns
   `Prediction(name, float(scores[i]))` where `scores` are logits (routinely > 1
   on raw180, e.g. 13.3). ADR-0005 added the calibration machinery
   (`TemperatureCalibrator` / `apply_calibration`) but deferred wiring it into the
   serving path because that needs the full per-class logit vector — which lives
   right here in the adapter. (FR-006-serve.)

Hard constraints (same frozen contract as ADR-0004/0005/0006):

- The `results.csv` columns are **frozen**; the raw `probability` must **not** be
  overwritten — a calibrated value rides only in the `detections.jsonl` sidecar.
- The Perch 2 backbone (`_prepare` / peak-norm / 32 kHz·5 s) is **not** touched.
- Any change that could move production output is **OFF by default** with a golden
  diff proving the default path is byte-for-byte unchanged.
- No module-level `import tensorflow` — the new math is pure numpy / file I/O.

A sharp edge called out by the silent-failure review and baked into the design:
the gate formula `clamp01(p_species·(1+p_bird·k))` operates on **softmax
probabilities**, but the default served value is a **logit**. So *the OFF state
must be a literal early-return of the raw logit, not the formula evaluated at
`k = 0`* — because `clamp01(softmax(scores)[i]·(1+p_bird·0)) = softmax(scores)[i]`
would silently swap the served logit for a probability while looking like a no-op.

## Decision

Two additive signals inside `Perch2Adapter.classify`, both OFF by default, both
computed from the SAME logits.

### 1. Presence soft-gate (FR-003)

- `_load_bird_mask()` reads `<model_path>/assets/perch_v2_ebird_classes.csv` once
  (cached, mirroring `_load_labels`): a boolean array, True where the row has an
  eBird code, False where it is `no_ebird_code`. Fail-open — a missing /
  unreadable / empty asset, or a mask whose length ≠ the logit count, logs a
  warning and disables the gate (raw logits) rather than crashing or
  mislabelling. The `len(mask) == len(scores)` cross-check is the load-bearing
  guard (the same one the label off-by-one uses); the header drop is conservative
  (only a known sentinel like `ebird2021`, since eBird codes are legitimately
  single-token, unlike the binomial label heuristic).
- `k = FAUN_PRESENCE_GATE_K` (`faun.settings.presence_gate_k`), parsed with
  `positive=False` so **`k = 0` is the valid OFF value**, not a rejected
  non-positive.
- **`k = 0`: literal early-return of `float(scores[i])`** — the raw logit,
  byte-for-byte unchanged. **`k > 0`** (with a valid mask): compute
  `softmax(scores)` once, `p_bird` = bird-mass via the unit-tested free function
  `bird_presence_mass(scores, mask)`, and return
  `p_final = clamp01(p_species·(1+p_bird·k))` per class. The boost is a per-segment
  uniform multiplier, so it rescales confidence by how clearly a bird is present
  without re-ranking within a segment.
- The math lives in free functions (`_softmax`, `bird_presence_mass`,
  `apply_presence_gate`) so it is unit-tested without TensorFlow, and `classify`
  calls them rather than re-inlining the formula.

### 2. Serve-time calibration (FR-006-serve)

- `Prediction` gains an **optional** `prob_calibrated: float | None = None`
  (additive: every `Prediction(species, probability)` call stays valid).
- `_load_calibrator()` loads a pickled `TemperatureCalibrator` from
  `PERCH_V2_CALIBRATOR_PATH` (`faun.settings.perch_v2_calibrator_path`), cached and
  fail-open (a bad/missing pickle → `None` → `prob_calibrated` stays `None`).
- When configured, `classify` fills `prob_calibrated` with
  `apply_calibration(calibrator, scores)[i] = softmax(scores / T)[i] ∈ [0, 1]`,
  computed from the **raw** logits, **independent of the presence gate**. The raw
  `probability` is never overwritten.
- `Label.from_prediction` threads the field through (`getattr(pred,
  "prob_calibrated", None)`), so `api._build_labels` propagates it into
  `detections.jsonl` with **no change to `api.py`** — the data-shape change is
  absorbed at the single `Prediction → Label` lift point.

## Consequences

Positive:

- The model's own bird-vs-noise knowledge can suppress confident-but-implausible
  birds on noisy segments, tunably, with one env var and zero extra inference.
- An honest `[0, 1]` calibrated probability is now available on the serving path
  (sidecar), without touching the raw score or the frozen CSV.
- Every honesty invariant is mechanically enforced: with `FAUN_PRESENCE_GATE_K = 0`
  and no calibrator the golden CSV + normalized `detections.jsonl` are byte-
  identical to `main@8c24ca9`; the adapter's k=0 raw-logit guarantee is pinned by
  a dedicated unit test (the golden gate runs a fake adapter, so the adapter-level
  guarantee is held by `test_classify_k0_returns_raw_logits`, not the golden gate
  itself).

Negative / costs:

- The gate is a per-segment confidence rescale, not a per-species presence model;
  it boosts/suppresses uniformly within a segment and does not re-rank.
- `k` and the calibration temperature `T` are operator-set; this wave ships the
  mechanism, not a fitted `k` or a deployed calibrator (calibration is only
  measurable on iNatSounds heads — domain shift, METRICS_HONESTY §10.4 — and `k`
  wants reserve audio to tune). Validating both on the cluster cache is a
  best-effort follow-up, not a gate.
- `prob_calibrated` carries `softmax(logits/T)` of the **served** logit; if the
  presence gate ever changed the served value this would diverge — by design it
  does not (calibration reads raw scores).

## References

- `faun/classification/perch_v2.py` — `_load_bird_mask`, `_load_calibrator`,
  `_softmax` / `bird_presence_mass` / `apply_presence_gate`, the gate +
  calibration logic in `classify`; `PERCH_V2_EBIRD_FILE`, `NO_EBIRD_CODE`.
- `faun/classification/__init__.py` — `Prediction.prob_calibrated`.
- `faun/detections.py` — `Label.from_prediction` threads `prob_calibrated`.
- `faun/settings.py` — `presence_gate_k` (`FAUN_PRESENCE_GATE_K`),
  `perch_v2_calibrator_path` (`PERCH_V2_CALIBRATOR_PATH`).
- `faun/retraining.py` — `TemperatureCalibrator`, `apply_calibration` (ADR-0005).
- `experiments/report/METRICS_HONESTY.md` §10.4 — the iNat-vs-raw180 domain-shift
  caveat any calibration/gate number must carry.
- ADR-0005 (calibration machinery), ADR-0004/0006 (the additive / OFF-by-default
  pattern).

## Review

Three read-only reviewers audited the implementation (artifacts in
`~/.claude/plans/wave3-audit-a{1,2,3}.md`):

- `silent-failure-hunter` → **SHIP**. Empirically confirmed the k=0 literal
  raw-logit path (a logit 13.27 would otherwise become softmax 0.79),
  `_env_float(positive=False)` accepting k=0, every fail-open path
  (missing/unreadable/empty asset, length mismatch, bad calibrator pickle), and
  that calibration reads raw scores independent of the gate. Clarified that the
  adapter's k=0 guarantee is held by a unit test, not the (fake-adapter) golden
  gate — coverage adequate.
- `architect` → **SHIP**. All four hard invariants hold; placement, lazy-loader
  mirroring of `_load_labels`, and the `Prediction → Label.from_prediction →
  detections.jsonl` flow without an `api.py` change are clean.
- `code-reviewer` → **SHIP-WITH-CHANGES**, 0 blockers. Flagged that
  `bird_presence_mass` was exported and unit-tested but not called by `classify`
  (re-inlined math) — integrated: `classify` now calls the free function so the
  test covers shipped code.

Integrated (HARDEN): `classify` routes `p_bird` through `bird_presence_mass`
(was an inline copy). Deferred (NICE): the local `_softmax` duplicates
`faun.retraining._softmax` — kept to keep `perch_v2` independent and TF-free.
