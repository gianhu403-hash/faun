# ADR 0006 — Honest segmentation: dense windows, silence filter, temporal NMS, smoothing

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/wave2-segmentation`

## Context

The served pipeline detects events with a **transient onset detector**
(`faun/ml/onset.py`, reused from the hackathon edge node). It was calibrated for
sharp impulses — chainsaw, gunshot, axe — and triggers on a sudden short/long
energy ratio. For *bird song*, which is often sustained and only modestly above
the noise floor, that detector under-recalls: a continuous warble may never
produce the spike the onset gate wants, so whole songs go unsegmented and the
classifier never sees them.

Three gaps follow from a transient-only segmenter, all fixable on CPU without
GPU or labels:

1. **No dense coverage** — sustained song between transients is missed entirely.
2. **No way to prune empties** — a recall-oriented grid would emit windows over
   silence.
3. **No overlap control beyond the legacy "onset inside the previous segment"
   drop** — a more aggressive grid needs a principled de-duplication.

Plus a fourth, output-side gap unrelated to windowing:

4. **Per-detection probabilities are noisy over time** — a single high score on
   one segment reads the same as a sustained run of high scores, even though the
   latter is far stronger evidence of presence.

Hard constraints (same frozen contract as ADR-0004 / ADR-0005):

- `SegmentExtractor.extract(waveform, sr)` and the `results.csv` columns
  `track,start_sec,duration_sec,species,probability` are **frozen**
  (`faun/INTERFACES.md`); `CsvWriter` is `extrasaction="raise"` — additive output
  is sidecar / env only.
- The clip↔detection **row-alignment** invariant (ADR-0003) — `run_batch`
  (`faun/pipeline.py`) builds one clip per segment in `extract()` order — must
  hold. Any de-duplication therefore has to happen **inside `extract()` before it
  returns**, never as a post-`run_batch` filter or reorder.
- Any change that could move production output must be **OFF by default** behind a
  flag, with a golden diff proving the default path is byte-for-byte unchanged
  (the frozen `tests/fixtures/golden/` baseline from `main@8c24ca9`).
- The onset detector's **public API** (`OnsetDetector`) must not change — it is
  also consumed by `legacy/edge` and `tests/test_onset.py`.
- The raw `probability` is **not** rewritten (ADR-0005); smoothing is sidecar-only.

A consequence worth stating plainly: `extract()` is **classifier-free** — it sees
only audio, never scores. So a *per-species* NMS is impossible at this layer by
construction; only a temporal NMS is available here.

## Decision

Four additive features, every one OFF / no-op by default.

### 1. Dense windows (FR-002) — `SegmentExtractor` constructor knobs

Keyword-only `dense_windows=False`, `window_s=5.0`, `hop_s=2.5`. When enabled,
`extract()` replaces the onset path with a fixed grid of `window_s` windows hopped
by `hop_s` over the recording (final window clamped to the file end, dropped if
shorter than `min_segment_s`). The onset path stays the default; the
`extract(waveform, sr)` signature is unchanged.

### 2. Silence pre-filter (FR-002b) — `silence_filter=False`

When enabled, a produced window whose **whole-window 16 kHz RMS** is at or below
`faun.ml.onset.MIN_ABSOLUTE_ENERGY` is dropped. The threshold is anchored to the
detector's own floor constant; the metric is whole-window RMS (a coarser, slightly
more aggressive empties filter than the detector's per-frame energy gate — it is
not an onset re-derivation), used mainly to prune all-silence dense windows.

### 3. Temporal-IoU NMS (FR-002c) — `nms_iou=None`

Generalizes the legacy construction-time overlap drop. `None` keeps the exact
legacy behaviour (an onset starting before the previous segment's end is skipped).
A float in `(0, 1]` adds a greedy temporal **intersection-over-union** suppression
on the built segments: since there are no scores to rank by, the deterministic,
alignment-safe rule is *keep the earlier segment, drop later ones whose IoU with a
kept one exceeds the threshold*. Suppression is strict (`IoU > threshold`), so an
exact-threshold overlap is retained. **Temporal only** — see the classifier-free
note above.

All three run inside `extract()` (dispatch order: generate → silence filter →
NMS) before the list is returned, so row-alignment is preserved with no
post-`run_batch` surgery.

### 4. Per-recording probability smoothing (FR-004) — `prob_smoothed.json` sidecar

`faun.output.prob_smoothed` / `write_prob_smoothed` derive a sidecar from the
in-memory `Detection` list after the streaming loop (output-only, mirroring
`species_presence`). For each `(recording, species)` the best-label probabilities
are ordered by segment start time and run through a centered, edge-clamped moving
average (default window 3). `run_pipeline` writes it **only when
`FAUN_PROB_SMOOTHING` is set** (`faun.settings.prob_smoothing`, default `False`);
unset → no file, default job directory unchanged. The raw `probability`,
`results.csv` and `detections.jsonl` are untouched.

### Scope note — segmentation knobs are primitives, prod wiring is deferred

This wave deliberately lands the dense/silence/NMS knobs as **`SegmentExtractor`
constructor parameters with direct-construction tests**, not an env/settings/CLI
toggle on the served `run_pipeline`. `run_batch` still defaults to a bare
`SegmentExtractor()` (all knobs off), so the served path is byte-identical. Wiring
these three through `faun.settings` into a configured extractor (the same seam
`prob_smoothing` already has) is an intentional, low-risk follow-up — kept out of
this PR to stay surgical and keep the golden surface minimal. Until then the recall
knobs are reachable for offline experiments and tests, not from the operator's
seat. (Raised by the architecture audit, F1 — accepted as a documented deferral.)

## Consequences

Positive:

- The segmenter can trade transient-only precision for sustained-song recall
  (dense grid), prune the resulting empties (silence filter), and de-duplicate
  overlap (temporal NMS) — three orthogonal, composable axes.
- Smoothing gives the biologist a denoised presence-over-time view without
  touching the raw score or the frozen CSV.
- Every honesty invariant is mechanically enforced: all knobs OFF → the golden
  CSV + normalized `detections.jsonl` diff is byte-identical to `main@8c24ca9`;
  alignment holds under dense+NMS (regression test); the onset public API and
  `test_onset.py` are untouched.

Negative / costs:

- The recall knobs have no operator seam yet (see the scope note) — the wave lands
  the capability, not the production toggle.
- `nms_iou` interacts with the dense grid's fixed geometry: on a `window_s`/`hop_s`
  grid the adjacent-window IoU is a constant the operator must reason about
  (e.g. a 5 s/2.5 s grid gives adjacent IoU = 1/3), rather than an absolute value.
- Smoothing collapses each `(recording, species)` to one time series; the window
  is a fixed moving average, not an adaptive or model-aware smoother.

## References

- `faun/segmentation/__init__.py` — `SegmentExtractor` (`dense_windows`/`window_s`/
  `hop_s`/`silence_filter`/`nms_iou`, `_dense_windows`/`_drop_silent`/
  `_temporal_iou`/`_nms_temporal`).
- `faun/ml/onset.py` — `MIN_ABSOLUTE_ENERGY` (the reused silence floor); public
  `OnsetDetector` unchanged.
- `faun/output/__init__.py` — `prob_smoothed`, `write_prob_smoothed`,
  `_moving_average`.
- `faun/api.py` — `run_pipeline` writes `prob_smoothed.json` (FR-004, gated).
- `faun/settings.py` — `prob_smoothing` (`FAUN_PROB_SMOOTHING`).
- `faun/pipeline.py` — `run_batch` (the row-alignment contract, ADR-0003).
- `tests/fixtures/golden/` + `tests/pipeline/test_golden.py` — the frozen
  byte-identity gate this rides on.
- ADR-0003 (row-alignment), ADR-0004/0005 (the additive / OFF-by-default pattern).

## Review

Four read-only reviewers audited the implementation (artifacts in
`~/.claude/plans/wave2-audit-a{1,2,3,4}.md`):

- `architect` → **SHIP-WITH-CHANGES**. All five hard invariants pass. Must-fix
  F1: the segmentation knobs have no operator seam — integrated as the documented
  deferral above. Tightened the `silence_filter` docstring (F2).
- `silent-failure-hunter` → **SHIP-WITH-CHANGES**. Confirmed OFF-by-default
  honesty, alignment preservation, and the window/NMS/RMS math. Soft blocker B1:
  the golden harness `run_golden` did not pin `FAUN_PROB_SMOOTHING` in its env
  neutralization — fixed (the file is the reused template for later waves).
- `code-reviewer` → **SHIP**. Strictly additive, mirrors the `species_presence`
  pattern; flagged only the `silence_filter` docstring wording (fixed).
- `prompt-engineer` → **SHIP**. All success criteria enforced by binary tests
  against the frozen baseline; suggested two edge tests (moving-average window >
  series, NMS exact-threshold boundary) — both added.
