# ADR 0009 — Demo polish: spectrogram PNGs, Raven export, review filters

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/wave5-demo-polish`

## Context

The review UI (`/review`) is the ornithologist's hand-off surface: it lists a
job's detections with the real audio clip and a relabel control. Three gaps make
it weaker than it should be for the pilot demo, all fixable without touching the
pipeline output:

1. **No visual.** An expert scrubbing dozens of clips by ear alone is slow; a
   spectrogram thumbnail lets them triage at a glance (and a sustained song vs a
   transient reads instantly from the image).
2. **No standard-tool export.** Ornithologists work in Raven Pro / Audacity;
   there is no way to hand them the detections as a selection table they can open
   directly.
3. **No filtering.** A long detection list cannot be narrowed to a species of
   interest or a confidence floor.

Hard constraints (same frozen contract as ADR-0004..0007):

- The `results.csv` columns and frozen signatures are unchanged; this wave adds a
  route, a CLI subcommand, static assets, and one pinned dependency — **no
  pipeline-output change**, so the golden gate is untouched.
- `faun/` must **not** import from `experiments/` (benchmark scaffolding, not a
  runtime dependency) — the spectrogram renderer has to be self-contained.
- No module-level heavy imports: `matplotlib` (like TF/torch) stays lazy.
- The clip-download route's hex-id traversal guard (`_DETECTION_ID_RE`) must be
  reused verbatim by any new per-detection file route.

## Decision

Three additive demo affordances.

### 1. Spectrogram PNG (FR-008)

- New top-level module `faun/spectrogram.py` — a self-contained copy of
  `experiments.common.save_spectrogram_png` plus the small framing helper it
  relied on, so the pipeline never imports `experiments/`. `matplotlib` (Agg,
  headless) is imported lazily inside the function; the dependency is pinned in
  `requirements-pipeline.txt` (`matplotlib==3.9.2`) so the deploy image has it
  but importing the module costs nothing.
- New route `GET /jobs/{id}/segments/{det}.png` mirrors `get_segment`: the same
  `_DETECTION_ID_RE` hex-id guard (applied before any path is built), the same
  404 shape. It renders the PNG **lazily** from the existing `segments/<id>.wav`
  on first request and **caches** it next to the clip (atomic temp-write +
  `Path.replace`, so a concurrent double-miss never serves a torn image); a
  render failure is a logged 500, never a worker crash. Because it renders only
  on HTTP request — after `run_pipeline` has returned — it never affects the
  pipeline output, and `export-clips` (which bundles only `.wav` `segment_path`s)
  never picks up the cached PNGs.
- `review.html` shows an `<img loading="lazy">` spectrogram thumbnail per
  detection, so the browser only fetches (and thus only triggers rendering of)
  images that scroll into view.

### 2. Raven / Audacity export (FR-009)

- New CLI `faun export-raven --job <dir> --out <tsv>` → `_export_raven`. Like
  `export-clips`, it parses `detections.jsonl` **generically** (no
  `faun.detections` import — it is a neutral format dump, not a provenance gate)
  and writes a Raven Pro selection table (TSV): one row per detection with
  `Begin Time (s)` = `start_s`, `End Time (s)` = `start_s + duration_s`, and
  `Species` = the **current** (most recent) label — matching the review UI's
  `currentLabel`, so a ranger's correction is what exports. `Low/High Freq` bound
  the classifier band (0–16 kHz) as an honest fixed default (the clip sr is not
  in the jsonl).

### 3. Client-side review filters (FR-010)

- A filter bar in `review.html`: a species-substring text box and a
  minimum-confidence number box. Pure client JS (no backend call) — each rendered
  row stashes its current species + numeric probability as data attributes;
  `applyFilters` toggles row visibility. A row whose current label has no numeric
  probability (e.g. a human correction) always passes the confidence filter (it
  is never noise). A relabel re-stashes the row's data and re-applies the filter.

## Consequences

Positive:

- The expert gets a visual-first review (spectrogram triage), a one-command
  hand-off into their existing tooling (Raven/Audacity), and quick narrowing of a
  long list — all without any pipeline or contract change.
- The spectrogram renderer is a reusable leaf module (numpy in → PNG out), usable
  from the CLI/tests, not coupled to FastAPI.

Negative / costs:

- A new runtime dependency (`matplotlib`) on the served image (the slim TF-free
  rollback image needs it too, hence the `requirements-pipeline.txt` pin).
- The filter bar is client-only JS, so it is verified by manual/visual check, not
  pytest (FR-010 is the lowest-priority item of the wave); the route and the
  Raven export ARE covered by tests.
- `Low/High Freq` in the Raven table is a fixed band, not a per-detection
  measured one (the jsonl carries no spectral bounds); the expert refines the box.

## References

- `faun/spectrogram.py` — `save_spectrogram_png`, `_frame` (self-contained copy).
- `faun/api.py` — `get_segment_spectrogram` (the `.png` route; mirrors
  `get_segment`, atomic render-cache).
- `faun/cli.py` — `export-raven` / `_export_raven` (generic JSONL, like
  `_export_clips`).
- `faun/static/review.html`, `faun/static/styles.css` — spectrogram `<img>` +
  filter bar.
- `requirements-pipeline.txt` — `matplotlib==3.9.2` (lazy + pinned).
- `experiments/common.py:save_spectrogram_png` — the original this copies (NOT
  imported).
- ADR-0004..0007 — the additive / OFF-by-default / frozen-contract pattern.

## Review

Three read-only reviewers audited the implementation (artifacts in
`~/.claude/plans/wave5-audit-a{1,2,3}.md`):

- `architect` → **SHIP**. All five invariants hold; `faun/spectrogram.py` is the
  right self-contained home, the lazy-render-and-cache is sound and pollutes
  nothing, `_export_raven` correctly mirrors the generic-JSONL `_export_clips`,
  the matplotlib boundary is clean. Flagged the non-atomic render (LOW).
- `silent-failure-hunter` → **SHIP-WITH-CHANGES**, 0 blockers. Confirmed the
  hex-id guard fires before any path construction, render failure surfaces as a
  logged 500 (never swallowed), spectrogram edge cases (empty/sr=0/NaN/short
  clip) all produce a valid PNG, Raven export is honest, and the filter never
  silently hides a probability-less row. Flagged the non-atomic render (LOW).
- `code-reviewer` → **SHIP**. Small, surgical, mirrors `get_segment` /
  `_export_clips`; tests assert real behaviour (PNG magic bytes, exact Raven
  times/species, traversal/missing-clip 404s); no XSS (`textContent`/`dataset`
  only; `innerHTML=""` is a clear).

Integrated (HARDEN): the PNG render now writes to a temp file then atomically
`Path.replace`s it (matching `write_detections`' tmp+replace idiom), closing the
concurrent-double-miss torn-PNG window both reviewers flagged. Deferred (NICE):
the Raven band is a fixed default; FR-010 stays a manual-check (client JS).
