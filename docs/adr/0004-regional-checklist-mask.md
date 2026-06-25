# ADR 0004 — Regional species allow-list + per-trap presence aggregate

- Status: Accepted
- Date: 2026-06-24
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/wave1-reserve-mask-presence`

## Context

Faun serves **zero-shot Perch 2** in production (`faun.antopkin.ru`): the model
picks among ~14 795 species with no fine-tuning on our fauna. Two gaps make the
output less defensible than it could be, and both are fixable on CPU, without
GPU and without a human-labelled corpus:

1. **No regional filter.** Zero-shot over a global 14 795-class head will, on
   noisy trap audio, occasionally return a *confident* species that does not
   occur anywhere near the deployment region. The honest measured number we
   have (held-out macro-F1 **0.834** on 50 species from an example Palearctic
   iNatSounds subset — not a validated reserve checklist — see
   `experiments/report/METRICS_HONESTY.md` §10) says the backbone separates
   *regional* species well; it says nothing about suppressing out-of-region
   false positives, which a checklist trivially can.

2. **No per-trap, per-day rollup.** The biologist's actual question is "which
   species was on trap A over this period, and how confidently" — but the
   pipeline only emits a flat per-segment `results.csv` and `detections.jsonl`.

Hard constraints this decision must respect:

- The `results.csv` columns `track,start_sec,duration_sec,species,probability`
  are **frozen** (`faun/INTERFACES.md`), and `CsvWriter` uses
  `extrasaction="raise"` — **no new CSV column is possible**; additive output
  must be a sidecar.
- The Perch 2 backbone (`_prepare` / peak-norm / 32 kHz·5 s) is **not touched**.
- Any change that could move the production numbers must be **OFF by default**
  behind an env flag until validated, with a golden-CSV diff proving the
  default path is byte-for-byte unchanged (honesty gate).
- `StubAdapter` already emits the species token `"unknown"`, so a reject status
  (a later wave) must use a different token.

A known sharp edge: iNatSounds folder names are `Genus_species` (underscore)
while Perch labels, the checklist and prod output use `Genus species` (space).
During the 50-species eval this mismatch silently zeroed the name mapping
("иначе было 0/50") — so name normalization and a coverage check are
load-bearing, not cosmetic.

## Decision

Two additive features, one env flag, OFF by default; both leave the frozen
contract and the production output untouched until explicitly enabled.

### 1. `MaskedClassifier` — regional allow-list (FR-001)

A protocol wrapper over `SpeciesClassifier` (`faun/classification/__init__.py`),
applied in `faun.api._build_classifier` — **not inside any adapter** — so it
works identically for the Perch 2 zero-shot head and a future probe head.

- It calls the wrapped classifier, then keeps only predictions whose species is
  on the allow-list, logging each dropped one as `masked_out=<species>`.
- Names are compared through `_species_key` (underscore→space, whitespace
  collapse, casefold), generalizing `scripts/.../_binomial`, so a `Genus_species`
  or mixed-case checklist still matches the `Genus species` model labels.
- **Fail-loud, fail-open coverage gate.** On the first `classify`, the mask
  compares the allow-list against the wrapped classifier's own label vocabulary
  (`vocab_provider`, e.g. `Perch2Adapter._load_labels`). If fewer than
  `coverage_floor` (default 0.5) of the allow-list names appear in that
  vocabulary — a name-format mismatch, a typo'd checklist, or labels that fell
  back to `species_<i>` — the mask logs a warning and **disables itself**
  (passes every prediction through) rather than silently emptying the CSV. With
  no `vocab_provider` (Stub/Perch-1/BirdNET/YAMNet) the gate can't run and the
  mask stays a no-op.
- Config: `FAUN_SPECIES_ALLOWLIST` (`faun.settings`) is a path to a checklist
  file, or the literal `default`/`reserve` for the bundled seed. **Unset →
  the bare classifier is returned and output is byte-for-byte unchanged.**
- Seed: `faun/data/reserve_checklist.txt` — 69 Palearctic forest binomials,
  copied from and kept in sync by hand with
  `scripts/extract_inatsounds_subset.py:RESERVE` (NOT imported from `scripts/`),
  intended to be refined by the reserve's ornithologist.
- `faun.api._classifier_source` unwraps `MaskedClassifier.inner` so
  `detections.jsonl` provenance still records the real backbone
  (`model:perch-v2`), not the wrapper.

### 2. `species_presence.json` — per-trap, per-day aggregate (FR-005)

`faun.output.species_presence` / `write_species_presence` derive a sidecar from
the in-memory `Detection` list after the streaming loop (output-only; the CSV /
streaming path is untouched). Per `(trap_id, day)` group it lists each species
with its detection count and the max / mean of contributing pseudo-label
probabilities. The recording **day** is parsed from the filename by reusing
`faun.ingest`'s single timestamp parser (`null` when unparseable).

Attribution is **one species per detection**: the best (highest-probability)
model label, with a human ground-truth label (confirmed/corrected) overriding
it. Each group's `n_detections` counts **every** detected event (so the group
totals sum to the number of lines in `detections.jsonl`, an honest count);
per-species `detections` cover only the attributed events, so the gap is exactly
the all-masked-out events.

## Consequences

Positive:

- Operators can restrict the served output to plausible regional species with a
  single env var and an editable text file — no model change, no GPU, no labels.
- Because the mask wraps the protocol (not the adapter), it composes with any
  current or future classifier and never touches the frozen Perch 2 path.
- The biologist gets a ready "which species, where, how often, how confident"
  rollup without any new CSV column or pipeline-loop change.
- The honesty invariants are mechanically enforceable: allow-list OFF → golden
  CSV diff is byte-identical; a misconfigured checklist degrades to a no-op with
  a warning instead of emptying the output.

Negative / costs:

- The mask is an **output filter over the adapter's top-k**, not a true argmax
  over allow-list-restricted logits (the protocol only exposes top-k
  predictions). A regional species ranked below the adapter's `top_k` for a
  given segment can still be missed; raising `top_k` or restricting logits is a
  future option, deliberately out of scope here to keep the change surgical.
- The seed checklist duplicates `RESERVE` and must be kept in sync by hand
  (documented in the file header).
- Presence attribution collapses a multi-label detection to one species; a
  detection genuinely containing two species is attributed to its top label
  only. This is the standard, defensible choice for a presence-per-day summary.

## References

- `faun/classification/__init__.py` — `MaskedClassifier`, `load_allowlist`,
  `_species_key` / `_binomial`, `RESERVE_CHECKLIST_PATH`.
- `faun/api.py` — `_build_classifier` (mask wiring), `_classifier_source`
  (unwrap), `run_pipeline` (writes `species_presence.json`).
- `faun/output/__init__.py` — `species_presence`, `write_species_presence`.
- `faun/settings.py` — `species_allowlist` (`FAUN_SPECIES_ALLOWLIST`).
- `faun/data/reserve_checklist.txt` — bundled default seed (69 species).
- `experiments/report/METRICS_HONESTY.md` §10 — the 0.834 number this rides on
  (and what it is NOT: not raw180 serving accuracy).
- `faun/INTERFACES.md` — frozen CSV columns + signatures (unchanged).

## Review

Two read-only reviewers audited the implementation:

- `code-reviewer` → **SHIP**. Verified all six constraints (OFF-by-default,
  frozen contract, no module-level TF/torch, source-tag unwrap, fail-open,
  count invariants); flagged only doc/log polish.
  Artifact: `~/.claude/plans/wave1-reserve-mask-agent-a1.md`.
- `silent-failure-hunter` → **SHIP-WITH-CHANGES**. Empirically confirmed both
  catastrophic failure modes (silently-empty CSV, corrupted provenance) are
  defended across underscores / typos / empty / missing / directory /
  `species_<i>` / None-or-raising vocab provider.
  Artifact: `~/.claude/plans/wave1-reserve-mask-agent-a2.md`.

Integrated (HARDEN): doc precision on the settings comment (top-k filter, not
argmax); multi-level `.inner` unwrap in `_classifier_source`; one-time warning if
the ingest day-parser import ever fails; atomic (tmp+rename) `species_presence`
write. Deferred (NICE): `masked_out` log volume at large scale (kept at INFO per
spec — only active when the opt-in mask is enabled).

