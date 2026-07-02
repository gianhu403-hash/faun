# ADR 0010 — Two-model routing: bird baseline vs. target-taxon gate

- Status: Proposed
- Date: 2026-07-03
- Deciders: Faun v2 pipeline team (design), owner sign-off needed on `tau`
- Supersedes: —
- Branch of record: — (design doc; not yet implemented — see Consequences)

## Context

The product's differentiator is **not** birds. The client-facing goal (recorded
at the very first scoping meeting, `tasks/regional-meeting-2026-05-08/`) is
detecting **secretive target mammals** — садовая соня (*Eliomys quercinus*) and
выхухоль (*Desmana moschata*) are the named examples. Birds are the
**baseline**: the only thing actually shipped in production today is zero-shot
Perch 2 over its 14795-class bird-heavy head (`faun.antopkin.ru`,
`faun-api:2.3.0-ml-20260624`).

The reason the target taxon isn't served is not architectural — it's data.
Nowhere in the project, on the cluster or in any open dataset, is there a single
labeled clip of the target species:

- `raw180` (177 GB, 1656 files, the actual trap recordings) has an empty `NOTE`
  column in every `A1`–`A4` `info.txt` — verified across all four traps, live,
  2026-07-03. No species labels of any kind exist for this audio.
- iNatSounds — the largest public bioacoustic corpus we have access to — lists
  садовая соня as a taxonomy entry (`val.json`, category id 5465,
  Gliridae/Rodentia) but with **zero** audio annotations in the val split.
  Выхухоль is **absent** from all 5569 categories entirely (checked both the
  `Desman` substring and the whole `Talpidae` family).
- ESC-50 (the routing/negative-calibration set, see below) has no target-mammal
  class — its animal classes are generic domestic/farm sounds (cat, dog, cow,
  sheep, pig, hen, rooster, crow, frog, insects, crickets).

So a second, target-taxon classifier cannot be trained or evaluated today —
that is a data-acquisition problem (ranger annotation of `raw180`, or a
few-shot xeno-canto bootstrap; `experiments/exp_e10.py` already implements the
latter and sits `SKIPPED` for exactly this reason), not something this ADR can
solve. What this ADR *can* do now, without any target-mammal label, is stop the
bird baseline from **silently mislabeling non-bird audio as a confident bird
species**, and give the pipeline a socket a target-taxon model can be plugged
into once one exists.

**Why this is answerable without mammal labels.** The routing question is
binary — "is this segment a bird, or not" — not "which species is this."
Perch 2's own 14795-class head already separates into **9706 bird** classes
(rows with an eBird code) and **5089 non-bird** classes (`no_ebird_code`, the
FSD50K noise rows: wind, vehicle, speech, …), so the softmax mass over the bird
columns (`p_bird`) is a signal Perch 2 already computes internally — ADR-0007
introduced it as `bird_presence_mass(scores, bird_mask)` and wired it into the
presence soft-gate. That gate *rescales* a species probability continuously; it
does not *reject* a detection. Calibrating a hard threshold `tau` on the same
bird/not-bird axis needs bird-vs-not-bird labels, which we already have
plenty of, from two sources that are NOT the missing target-mammal data:

- ESC-50 (2000 labeled clips, 50 classes: `chirping_birds`/`hen`/`rooster`/`crow`
  are bird-adjacent, the other 46 are not).
- iNatSounds (birds: 3846 of 5569 categories are Aves).

This axis is already empirically validated: `experiments/exp_esc50_probe`
measured bird-vs-not-bird routing at **ROC-AUC 0.9962 (Perch)** /
**0.9969 (YAMNet)** on ESC-50. That is the evidence base for `tau`
calibration — it says the *bird/not-bird* signal is strong, it says **nothing**
about target-mammal recall, because no target-mammal audio was in that test.

Hard constraints (the same frozen contract as ADR-0004…0008):

- `results.csv` columns stay frozen; routing decisions live only in the
  `detections.jsonl` sidecar (`Label.status`).
- The Perch 2 backbone and `Perch2Adapter._prepare` are untouched.
- OFF by default, with a golden diff proving the default path is byte-for-byte
  unchanged — same pattern as ADR-0004/0006/0007/0008.
- No new module-level `import tensorflow` / `import torch`.

A caveat this ADR must state plainly, because it bears on any future
target-taxon model built behind this router: `p_bird` is computed from audio
that already passed through `faun.pipeline.to_classifier_input`, which resamples
every clip to `CLASSIFY_SR = 16000` Hz **before** any classifier — including
Perch 2 — sees it (`faun/pipeline.py:51,67-75`). `Perch2Adapter._prepare` then
resamples that already-16 kHz signal *up* to its native 32 kHz
(`faun/classification/perch_v2.py:239-247`), but upsampling cannot restore
content above the original 8 kHz Nyquist limit. So both the routing signal
(`p_bird`) and any future target-taxon classifier plugged into this router see,
at most, audio content up to ~8 kHz — regardless of what the field trap
actually recorded at 48 kHz. Whether садовая соня / выхухоль vocalize within
that band is an open question for a biologist (Q2 in
`docs/ОТВЕТЫ-НА-ВОПРОСЫ-2026-07-02.md`), not something this ADR resolves.

## Decision

Introduce `faun.classification.RoutingClassifier`, a wrapper `SpeciesClassifier`
that sits in front of the bird baseline and makes one binary call per segment:
"confidently a bird" → pass the baseline's prediction through; "not confidently
a bird" → mark it `STATUS_REJECTED` instead of forcing a bird species name onto
it. No target-taxon model is defined or trained by this ADR — the rejected
bucket is a labeled **parking lot**: today it means "route to the human review
queue" (`/review`, already shipped); tomorrow, once a target-taxon classifier
exists (blocked on data — see Context), it is the exact socket that model plugs
into instead of a human queue, with no change to the plumbing described here.

### 1. Expose `p_bird` on `Prediction` (additive)

- `Prediction` gains an optional `p_bird: float | None = None` field, following
  the exact precedent of `prob_calibrated` (ADR-0007): every existing
  two-argument `Prediction(species, probability)` call stays valid.
- `Perch2Adapter.classify` computes `p_bird` via the existing
  `bird_presence_mass(scores, bird_mask)` (ADR-0007, unchanged) **independent of
  the presence-gate flag** `FAUN_PRESENCE_GATE_K` — the gate reshapes
  `probability`; `p_bird` is a separate, always-available diagnostic once the
  bird-mask asset loads. If the mask fails to load (missing/unreadable/empty
  asset, length mismatch — the same fail-open path ADR-0007 already has),
  `p_bird` stays `None` and routing is disabled for that segment (fail-open, not
  fail-closed — an unrouted segment reverts to today's baseline behavior).
- `Label.from_prediction` threads `p_bird` into the `detections.jsonl` sidecar
  the same way it already threads `prob_calibrated` — additive, no `api.py`
  change at that lift point.

### 2. `tau`, calibrated on the bird/not-bird axis

- `FAUN_ROUTING_TAU` (`faun.settings.routing_tau: float | None = None`),
  parsed like `presence_gate_k` — but here `None`/unset is the OFF value (not
  `0.0`, since `p_bird = 0` is itself a meaningful "definitely not a bird"
  reading and must not collide with the disabled state).
- `tau` is fit offline against ESC-50 + iNat bird/not-bird labels — the same
  data `exp_esc50_probe` already used to report ROC-AUC 0.9962/0.9969 — by
  picking an operating point on that ROC curve. **No target-mammal label is
  needed to fit `tau`**, because the question it answers ("bird or not") does
  not require knowing which non-bird class a segment belongs to.
- **What `tau` cannot tell us**: the *cost* of a false reject (a real bird
  routed to the reject bucket) vs. a false accept (a non-bird — including a
  genuine target-mammal call — kept in the bird bucket and never surfaced for
  review) is a product decision, not a statistic. This is `docs/ОТВЕТЫ-НА-ВОПРОСЫ-2026-07-02.md`
  Q6, owner-only, unresolved by this ADR.

### 3. `RoutingClassifier`

- `RoutingClassifier(inner: SpeciesClassifier, tau: float | None)` wraps any
  `SpeciesClassifier`. `classify(segment, sr)` calls `inner.classify(segment,
  sr)`, reads `p_bird` off the top prediction, and:
  - `tau is None`, or `p_bird is None` (mask unavailable) → return `inner`'s
    predictions **unchanged** — the wrapper is a transparent passthrough. This
    is the literal-early-return pattern ADR-0007 established for `k = 0`, here
    applied to `tau = None`.
  - `p_bird >= tau` → return `inner`'s predictions unchanged (confidently a
    bird — the shipped baseline owns this segment).
  - `p_bird < tau` → the caller (at the `Label.from_prediction` lift point in
    `faun/api.py`/`faun/labeling`) uses `status=STATUS_REJECTED`
    (`faun/detections.py:71` — already defined by ADR-0005/FR-006, currently
    unused outside its own unit test) instead of `STATUS_PSEUDO`. The species
    field is **not** blanked or replaced with a fabricated "not a bird" label —
    the baseline's raw prediction stays visible in the sidecar for audit, only
    its lifecycle status changes, so a reviewer can see exactly what the bird
    model guessed and why it was set aside.
- `RoutingClassifier` does not construct or call a second (target-taxon)
  classifier — there is not one to call. If/when one exists, it is expected to
  sit where the reject branch currently hands off to the human queue: the
  interface point is `status=STATUS_REJECTED` detections, already a
  first-class, queryable bucket (`is_ground_truth`/status filtering exists in
  `faun/detections.py` today).

### 4. OFF by default

- `FAUN_ROUTING_TAU` unset (`None`) → `faun.api._build_classifier` never
  constructs a `RoutingClassifier`; the served classifier is exactly what it is
  today. Golden diff proves this byte-for-byte, same invariant as
  ADR-0004/0006/0007/0008.
- Setting `FAUN_ROUTING_TAU` only ever **adds** `STATUS_REJECTED` detections
  where the baseline used to emit `STATUS_PSEUDO`; it never changes
  `probability`, `prob_calibrated`, or any CSV column.

## Consequences

Positive:

- Gives the pipeline an honest "this doesn't look like a bird" outcome using a
  signal (`p_bird`) and a status value (`STATUS_REJECTED`) that already exist
  and are already unit-tested — this wave would be almost entirely wiring, no
  new inference, no GPU, no target-mammal data required.
- Separates two decisions that are currently conflated in ADR-0007's soft gate:
  "how much should this bird-species probability be trusted" (continuous
  rescale, stays as-is) vs. "should this be treated as a bird prediction at
  all" (a binary lifecycle decision, new). A reviewer gets a `rejected` queue
  to look at instead of a long tail of low-probability bird guesses mixed in
  with everything else.
- Prepares the exact socket a target-taxon model needs once one is trainable —
  no re-architecture the day ranger-labeled `raw180` data or a xeno-canto
  bootstrap (`experiments/exp_e10.py`) produces target-mammal embeddings.

Negative / costs:

- **This ADR ships zero target-mammal recognition.** The reject bucket is not
  "identified as садовая соня/выхухоль" — it is "not confidently a bird,"
  which could be a target mammal, an unrelated animal, wind, or silence. Do not
  present a `STATUS_REJECTED` count as a mammal-detection count.
- `tau` is a hyperparameter this ADR proposes a *fitting procedure* for
  (ESC-50/iNat ROC operating point) but does not fit or ship a value — and the
  underlying error-cost trade-off (Q6) is owner-only.
- The 8 kHz effective ceiling (`CLASSIFY_SR = 16000` before any classifier,
  including the router, ever sees the audio) applies to `p_bird` itself, not
  just to a hypothetical future target model. If target vocalizations sit above
  8 kHz, this router — like the rest of the pipeline — is architecturally blind
  to them regardless of `tau`. Confirming the vocalization band is a biologist
  question, not resolved here.
- Fail-open on a missing bird-mask means routing silently disables itself
  rather than crashing — good for uptime, but means an operator must actually
  check logs to know whether routing is active, not just whether
  `FAUN_ROUTING_TAU` is set.
- Not yet implemented: no branch, no tests, no `RoutingClassifier` class exists
  in `faun/classification/` as of this ADR (verified: `grep -rn
  RoutingClassifier faun/` is empty at HEAD `9ccad2a`). This document is the
  design to review before a code wave picks it up.

## References

- `faun/classification/perch_v2.py:96` — `bird_presence_mass` (reused, not
  duplicated), `:108` `apply_presence_gate` (ADR-0007, unchanged).
- `faun/detections.py:41,71` — `STATUS_REJECTED` (defined by ADR-0005/FR-006,
  currently exercised only by `tests/test_detections.py`, not wired into any
  runtime decision — this ADR proposes its first real producer).
- `faun/pipeline.py:51,67-75` — `CLASSIFY_SR = 16_000`, `to_classifier_input`,
  the mandatory pre-classifier downsample that caps `p_bird`'s effective
  bandwidth at ~8 kHz.
- `faun/settings.py:224,274` — `presence_gate_k` / `_env_float(...,
  positive=False)`, the pattern `routing_tau` would follow (with `None`, not
  `0.0`, as the OFF sentinel).
- `experiments/exp_esc50_probe` — ROC-AUC 0.9962 (Perch) / 0.9969 (YAMNet)
  bird-vs-not-bird on ESC-50, the evidence base for calibrating `tau`.
- `experiments/exp_e10.py` — few-shot садовая соня/выхухоль prototype matching
  against xeno-canto references; `SKIPPED` today for lack of reference audio,
  the closest existing code to a future target-taxon model this router would
  hand off to.
- `docs/ОТВЕТЫ-НА-ВОПРОСЫ-2026-07-02.md` Q1 (no target-mammal label anywhere),
  Q2 (vocalization band, owner/biologist question), Q6 (error-cost trade-off,
  owner-only).
- ADR-0005 (`STATUS_REJECTED` origin, calibration machinery), ADR-0007 (`p_bird`
  origin, the presence soft-gate this proposal complements rather than
  replaces), ADR-0004/0006/0008 (the additive / OFF-by-default pattern this
  proposal follows).

## Review

Not yet reviewed — this ADR is `Proposed`, written ahead of any implementation
branch, specifically so the design (in particular: reusing `p_bird`/
`STATUS_REJECTED` rather than inventing new fields, and the explicit non-claim
of target-mammal recognition) can be checked before code is written. Update
`Status` to `Accepted` and fill in `Branch of record` once an implementation
wave picks this up and passes the same read-only audit gate as ADR-0004/
0006/0007/0008 (silent-failure / architecture / simplicity review).
