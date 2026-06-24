# ADR 0008 — Prototypical probe with a negative (background) class

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/wave4-probe`

## Context

The served probe head — `PerchProbeAdapter` over Perch 2's 1536-dim embedding —
is trained by `faun.retraining.train_probe_cv`, which fits an sklearn
`LogisticRegression`. That is a **closed-world** classifier: softmax over the
trained bird species sums to 1, so *every* input is forced onto one of them. An
out-of-distribution embedding — wind, a vehicle, speech, silence, or simply a
bird species the probe was never trained on — therefore maps to one of the known
birds, often *confidently*. On noisy trap audio this is exactly the failure mode
the reserve's biologist would distrust: a crisp species name on a segment that
contains no bird at all.

ADR-0007 already added one defence from the **zero-shot** path — the presence
soft-gate, which uses Perch 2's own 9706-bird / 5089-non-bird head to down-weight
predictions on segments with little bird-mass. But that gate lives in
`Perch2Adapter`, reads the model's native logits, and does nothing for the
**probe** path: a `LogisticRegression` probe has no non-bird logits and no "none
of the above" column. The probe needs its own OOD bucket.

Hard constraints (the same frozen contract as ADR-0004…0007):

- The `PerchProbeAdapter` (`faun/classification/perch_probe.py`) is **not
  touched**. The new probe must be a drop-in for the existing adapter — it must
  satisfy the `predict_proba(X) -> (N, C)` + `classes_` contract the adapter
  relies on and must pickle/unpickle cleanly through `save_probe` / `load_probe`
  and `YAMNetAdapter._load_probe`.
- The `results.csv` columns and every `faun/INTERFACES.md` signature stay frozen;
  this is **additive only** (a new trainer beside `train_probe_cv`, a new module
  constant, a new opt-in CLI flag in `scripts/eval_inatsounds_perchv2.py`).
- No module-level `import tensorflow` / `import torch`. The new math is pure
  numpy.

## Decision

Add `faun.retraining.train_prototype_probe` **beside** `train_probe_cv` (the
LogReg trainer is unchanged), returning a small picklable `_PrototypeProbe`.

### 1. Prototypical (nearest-centroid) probe

- `_PrototypeProbe.fit(X, y, *, negatives=None)` stores **one prototype per
  class** — the mean embedding of that class's training rows. Classes are sorted
  so the column order is deterministic and matches `classes_`. State is pure
  numpy (`classes_`, `prototypes_`, `metric`, `temperature`, `n_features_in_`),
  with no closures or third-party objects, so it pickles exactly like the in-repo
  `_ConstantProbe` and loads through the unchanged adapter.
- `predict_proba(X)` returns `softmax(similarity_logits / T)`, so each row sums
  to 1 and lies in `[0, 1]` — the contract `PerchProbeAdapter` and `YAMNetAdapter`
  require. `metric="cosine"` (default) L2-normalizes embeddings and prototypes
  and uses the dot product (cosine geometry, robust to embedding norm);
  `metric="euclidean"` uses the negative squared distance. `predict(X)` is the
  argmax label, aligned with `classes_`.
- The probe is **dimension-agnostic** (`n_features_in_` is learned at fit time),
  so it works for Perch 2's 1536-dim embedding and for small synthetic dims in
  unit tests alike. It is **deterministic** — same input gives the same
  prototypes and the same predictions, no random seed needed (centroids, not an
  iterative optimiser). Chosen over an `MLPClassifier` wrapper precisely for this:
  deterministic, interpretable, no torch, trivial to pickle.

### 2. Negative (background) class

- `NEGATIVE_CLASS = "__negative__"` is a module constant — a distinct *class* the
  probe predicts, deliberately different from the `"unknown"` species token
  (`StubAdapter`) and from `STATUS_REJECTED` (a label-lifecycle status, ADR-0005).
- `train_prototype_probe(X, y, *, negatives=None, ...)`: when `negatives`
  (`(M, D)` non-bird / background embeddings) is given, a distinct
  `NEGATIVE_CLASS` prototype is added (the mean of the negative embeddings), so an
  OOD embedding near it is classified negative rather than as a confident bird.
  When `negatives is None` the trainer behaves as a plain multiclass prototypical
  probe (no negative column) — a clean default.
- The served adapter is unchanged: `__negative__` simply rides the existing
  `classes_` naming path, so a query near the negative prototype surfaces
  `Prediction("__negative__", p)` as the top prediction. Operators can then map
  it to a reject status downstream (or filter it out) without any adapter change.

### 3. Head-to-head measurement (additive, cluster-only)

- `scripts/eval_inatsounds_perchv2.py` gains an **opt-in** `--prototype` flag (and
  `--negatives-from <dir>` for the negative class). When set, it trains the
  prototype probe on the **same disjoint train split** the LogReg probe uses and
  reports its held-out macro-F1 on the **same disjoint val** — apples-to-apples
  with the 0.834 LogReg number — and writes `prototype_heldout_macro_f1` into the
  JSON summary. Leakage guards are unchanged: fit on train only, measure on the
  disjoint val (the `#H2` invariant). Without the flag the script's default
  behaviour and output are byte-for-byte unchanged. iNatSounds val is birds-only,
  so the negative class is an OOD *mechanism* here, not a contributor to the
  reported macro-F1.

## Consequences

Positive:

- The probe path gains an explicit "none of the birds" outcome on noisy / OOD
  audio, with no GPU, no torch, no adapter change, and no extra inference — the
  negative prototype is just one more centroid.
- A deterministic, interpretable probe (centroids you can inspect) that is a
  drop-in for the existing served adapter and the existing
  `save_probe`/`load_probe` pickle path.
- The head-to-head is honest and apples-to-apples: same disjoint split, same val,
  same caveats as the LogReg number.

Negative / costs:

- This wave ships the **mechanism**, not a deployed probe. Whether the
  prototypical probe **beats** the 0.834 LogReg baseline (SC-D2) is a cluster
  measurement that is **not run here** — it needs Perch 2 embeddings of
  iNatSounds on the cluster (and non-bird clips for the negative class). The
  prototype probe is not wired into production; `train_probe_cv` remains the
  default trainer.
- The 0.834 baseline it is compared against is **iNatSounds held-out macro-F1**,
  not raw180 serving accuracy: iNat embeddings are the first 5 s of a focal clip
  (`fit_window` left-crop) while production feeds an onset-detected segment — a
  whole-clip-vs-onset domain shift. See `experiments/report/METRICS_HONESTY.md`
  §10.3–10.4. raw180 species accuracy remains unmeasured (needs ornithologist
  labels).
- A nearest-centroid probe assumes roughly unimodal, comparably-scaled clusters;
  a class with strong sub-modes (e.g. song vs call) is compressed to one centroid.
  `metric="cosine"` mitigates norm differences; richer prototypes (per-subcluster,
  or an `MLPClassifier`) are a documented future option, out of scope here.
- The negative class is only as good as the background embeddings fed to it: a
  narrow noise set under-covers the OOD space. Curating reserve background audio
  is a follow-up.

## References

- `faun/retraining.py` — `NEGATIVE_CLASS`, `_PrototypeProbe` (fit / predict_proba
  / predict / `_l2_normalize` / `_logits`), `train_prototype_probe`; reuses the
  module-level `_softmax`. `train_probe_cv` / `_ConstantProbe` / `save_probe` /
  `load_probe` unchanged.
- `faun/classification/perch_probe.py` — the **unchanged** `PerchProbeAdapter`
  the prototype probe is a drop-in for (`predict_proba` / `classes_`).
- `scripts/eval_inatsounds_perchv2.py` — `--prototype` / `--negatives-from`,
  `_prototype_head_to_head`, `_embed_negatives` (additive; default path unchanged).
- `tests/test_retraining.py`, `tests/test_perch_probe.py` — shape / sums /
  centroid / negative-class / dimension-agnostic / determinism / save-load
  round-trip / adapter drop-in.
- `experiments/report/METRICS_HONESTY.md` §10 — the 0.834 number and what it is
  NOT (not raw180 serving accuracy).
- ADR-0005 (`STATUS_REJECTED`, the calibration machinery), ADR-0007 (the
  zero-shot presence gate this complements on the probe path), ADR-0004/0006 (the
  additive / OFF-by-default pattern).

## Review

Read-only reviewers audited the implementation; artifacts (if any) live under
`~/.claude/plans/`. The integration of any blocking finding is recorded here once
the audit wave completes (HARDEN → edit + reference; NICE → deferred note). This
wave ships the trainer + tests + docs; the cluster head-to-head (SC-D2) and a
deployed probe are explicit follow-ups, not part of this change.
