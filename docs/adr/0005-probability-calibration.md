# ADR 0005 — Probability calibration (temperature scaling) + reject status

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/fr006-calibration`

## Context

The served zero-shot `Perch2Adapter` reports **raw logits** in the `probability`
field of every prediction (`faun/classification/perch_v2.py:classify` returns
`Prediction(name, float(scores[i]))` where `scores` are the model's logits). On
real reserve audio (raw180) these values are routinely **> 1** (e.g. 13.3, 12.5)
— a logit is not a probability, and a field literally named `probability`
carrying a logit misleads anyone reading the CSV or the review UI. This was
confirmed live while validating Wave 1's `species_presence.json` on the
`ae27d957` raw180 job.

Constraints this must respect (same frozen contract as ADR-0004):

- The `results.csv` columns `track,start_sec,duration_sec,species,probability`
  are **frozen** and `CsvWriter` uses `extrasaction="raise"` — no new CSV column.
- The raw `probability` must **not** be overwritten (provenance: the CSV keeps
  the model's own score; the negative list in the working plan forbids rewriting
  it and forbids reusing the `unknown` species token for a reject outcome).
- No behaviour change to the deployed output unless explicitly enabled
  (honesty gate); nothing may invent a calibrated number on its own.
- raw180 has **no human species labels**, so calibration cannot be fit or
  validated against reserve serving data yet — only against iNatSounds
  (the existing disjoint 50-species held-out), with the domain-shift caveat of
  `experiments/report/METRICS_HONESTY.md §10.4`.

## Decision

Add a small, dependency-free calibration layer in `faun/retraining.py` and the
sidecar plumbing to carry its output, all additive and OFF by default.

1. **Temperature scaling** (Guo et al. 2017), the standard calibrator for neural
   logits and the natural fix for "logits as probability":
   - `TemperatureCalibrator(temperature, classes)` — `apply(logits) → softmax(logits / T)`.
     `T = 1.0` is plain softmax (the un-fit / neutral state).
   - `fit_temperature(logits, y, *, classes=None)` — fits the single scalar `T`
     by minimising multiclass NLL (`scipy.optimize.minimize_scalar`, bounded);
     a single class or an optimiser failure falls back to `T = 1.0` (never
     raises — calibration must not crash a job).
   - `apply_calibration(calibrator, logits)` — **identity pass-through when
     `calibrator is None`**: with nothing configured the raw scores are returned
     unchanged, so the system never fabricates a calibrated probability.
   - `expected_calibration_error(probs, y_true, *, classes, n_bins)` — top-1
     confidence ECE, for honest before/after reporting.
   `scikit-learn` and `scipy` are already pinned; no new dependency. (Per-species
   Platt scaling is a deliberate future option, not built here.)

2. **Reject status** `STATUS_REJECTED = "rejected"` in `faun/detections.py`, a
   *lifecycle status* for a model abstain / negative-class outcome — distinct
   from the human `pseudo`/`confirmed`/`corrected` lifecycle, never ground truth
   (`is_ground_truth` stays `False`), and deliberately **not** the StubAdapter's
   `"unknown"` species token (which is a species name, not a status).

3. **Calibrated-probability sidecar field** `Label.prob_calibrated: float | None`
   (default `None`), additive in `to_dict`/`from_dict` (old `detections.jsonl`
   without the field loads as `None`). It lives **only** in `detections.jsonl`,
   never in the frozen CSV; the raw `probability` is left untouched.

## Consequences

Positive:

- The mechanism to turn the misleading raw logit into an honest `[0, 1]`
  probability now exists, validated against a real held-out set, without
  touching the frozen CSV or the raw score.
- `prob_calibrated` and `rejected` give downstream UI / export a calibrated
  number and an explicit abstain state to work with.
- Zero new dependencies; calibration code is pure numpy/scipy and TF-free, so it
  unit-tests without the cluster.

Negative / costs and what is deferred:

- **Serve-time wiring is not included here.** Populating `prob_calibrated` in the
  live pipeline needs the *full* per-class logit vector to softmax, but
  `SpeciesClassifier.classify` only returns the top-k `Prediction`s. Exposing the
  full logits from the adapter (or computing `prob_calibrated` inside it) is a
  follow-up; until then `prob_calibrated` stays `None` in production and the
  deployed output is unchanged.
- **Domain shift.** Any calibration fit on iNatSounds characterises iNat clip
  heads (first 5 s, `fit_window` left-crop), **not** raw180 onset segments.
  Provenance must say "calibrated on iNat heads", and transfer to raw180 serving
  stays unvalidated until the ornithologist labels raw180
  (`METRICS_HONESTY.md §10.4/§10.5`).

## References

- `faun/retraining.py` — `TemperatureCalibrator`, `fit_temperature`,
  `apply_calibration`, `expected_calibration_error`, `_softmax`.
- `faun/detections.py` — `STATUS_REJECTED`, `Label.prob_calibrated`.
- `faun/classification/perch_v2.py:classify` — the raw-logit `probability` this
  calibrates.
- `scripts/eval_inatsounds_perchv2.py` — disjoint held-out used for ECE
  validation.
- `experiments/report/METRICS_HONESTY.md §10.4` — domain-shift caveat.
- ADR-0004 — the additive / OFF-by-default / frozen-contract pattern this follows.
