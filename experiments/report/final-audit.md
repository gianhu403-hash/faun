# Faun pipeline — adversarial nightly audit

Scope: `faun/` (ingest, ordering, segmentation, classification, output, jobs,
storage, api.py, cli.py, static/index.html, ml/*), `tests/pipeline/`,
`experiments/`. Read-only. `legacy/` and `.github/` excluded.

Test run: `python -m pytest tests/ -q` → **177 passed in 3.14s**.
Real chain smoke (manually, outside suite): `run_pipeline` on a synthetic
trap folder produced correct `results.csv` + `results_meta.json`. The wiring
works; the suite just never exercises it (see MAJOR-2).

## Verdict: SHIP-WITH-CHANGES

Code is clean, well-documented, tests have real negative cases. No blockers.
Three majors worth fixing before this carries production traffic, plus minors.

---

## MAJOR

### MAJOR-1 — non-atomic manifest write + read race; duplicates the atomic `faun.jobs`
`faun/api.py:64-67` (`write_manifest` uses `path.write_text`, truncate+write,
no temp+rename) and `faun/api.py:144-158` (`_execute_job` writes status while
`GET /jobs/{id}` reads). `faun/api.py:57-61` (`read_manifest` → `json.loads`).

In production (uvicorn), the sync background task runs in the anyio threadpool
concurrently with `GET /jobs/{id}`. A reader can hit the file mid-truncate and
get partial/empty JSON → `json.JSONDecodeError` → unhandled 500 on a routine
poll. Meanwhile `faun/jobs/__init__.py:53-64` already implements exactly the
atomic write (`tmp + os.replace`) and a full Job lifecycle — but `api.py` never
imports it. So there are two job stores: an atomic one nobody uses and a racy
one in the hot path.

Fix: route the API job store through `faun.jobs` (atomic `os.replace`), or at
minimum make `write_manifest` write to `.manifest.json.tmp` then `os.replace`.
Eliminates both the race and the duplication.

### MAJOR-2 — the cross-wave integration seam has zero tests
`tests/pipeline/test_api.py:21-37` and `tests/pipeline/test_cli.py` both
monkeypatch `run_pipeline` (the only place ingest↔ordering↔segmentation↔
classification↔output are wired, `faun/api.py:82-136`). Every other test
exercises one module in isolation. Result: the actual contract glue
(`ordering.sort_entries(ingest.scan(...))`, `extractor.extract(waveform, sr)`,
`classifier.classify(segment, sr)`, dict-row → `_coerce_row`,
`CsvWriter().open(..., meta=...)`) is never run by the suite. I verified it
works by hand, but a signature drift in any wave would land green.

Fix: one end-to-end test that builds a tiny synthetic trap folder (as the
smoke I ran does) and calls the *real* `run_pipeline` with `StubAdapter`,
asserting the CSV header + at least one row + sidecar. ~20 lines, closes the
seam.

### MAJOR-3 — multi-trap runs mislabel the sidecar
`faun/api.py:112-118`: `TrapMeta` takes `trap_id`, `lat`, `lon` from
`manifest.entries[0]` only, but `files=[all entries]` and the CSV rows carry
`track=entry.trap_id` for *every* trap. `ingest.scan` returns multiple traps
when the path holds subfolders A1..A5 (`faun/ingest/files.py:397-399`), and the
CLI help explicitly documents this input ("source directory (one folder per
trap)", `faun/cli.py:19`). So a normal `faun process <dataset>` writes one
`results_meta.json` claiming the first trap's id/coords while the CSV spans all
traps — wrong provenance.

Fix: either (a) document/enforce one-trap-per-job and reject multi-trap input,
or (b) emit per-trap sidecars / a list of traps in the sidecar. Pick one; the
current half-state silently attributes everything to A1.

---

## MINOR

### MINOR-1 — YAMNet class-map download: bare except + no timeout, silent fallback
`faun/ml/yamnet.py:69-83`: `_load_yamnet_class_names` does `urllib.request.urlopen`
on a GitHub `master` URL (no timeout) inside `try/except Exception:` that sets
`_yamnet_class_names = []` with no log. On network failure `_classify_base_yamnet`
silently returns `unknown`/`background` (`:104-106`) — a degraded classifier
with no trace. Add a timeout and `logger.warning` on failure; ideally vendor the
class map instead of fetching `master` at runtime.

### MINOR-2 — onset `frame_index` is the peak frame, not the trigger frame
`faun/ml/onset.py:131-138`: `best_frame` tracks max-ratio, but trigger fires at
the *first* frame ≥ threshold. `segmentation` uses `event.frame_index` to place
the onset (`faun/segmentation/__init__.py:136`), so the reported onset time can
sit slightly after the true transient. Within the ±0.5 s test tolerance, but
the position is biased. If precise onset matters, return the trigger frame.

### MINOR-3 — UI polls forever on a transient server error
`faun/static/index.html:118-120,149-150`: `poll`/`loadResults` do `if (!res.ok)
return;` — a 500/404 on a poll silently no-ops and the 2 s interval keeps firing
with no user feedback. Show an error and `clearInterval` after N consecutive
failures.

### MINOR-4 — UI never sends `url`, only `source_path`
`faun/static/index.html:88` always packs the single field as `source_path`. The
contract mentions "folder/URL" and the API accepts either, so it's functionally
fine (API coalesces via `.source`), but the `url` branch of `JobRequest` is dead
from the UI's side. Cosmetic.

### MINOR-5 — naïve CSV split in the results table
`faun/static/index.html:152`: `r.split(",")` breaks on any field containing a
comma. Bird scientific names don't, so low risk, but a quoted-field species
would shift columns.

### MINOR-6 — `embed` private-symbol coupling across the experiments↔faun seam
`experiments/wrappers/yamnet_probe.py:18` and `faun/classification/yamnet.py:102`
both import `faun.ml.yamnet._load_models` (underscore-private). Works, but two
consumers now depend on a private symbol; promote it to a public loader.

### MINOR-7 — `faun/ml/datasphere_client.py` appears unused in scope
No importer under `faun/` or `experiments/` (grep clean). May be live in
`legacy/` (out of scope) — confirm before deleting; if orphaned, remove. The
`predictions[idx]` path also assumes a list and would `KeyError` on a dict
response, swallowed by the broad `except` at `:49` (logged, returns None — OK).

## Notes (not findings)
- `faun.jobs` is fully and well tested (atomicity, isolation, corrupt-manifest,
  invalid-status) — it's just unused by the API (see MAJOR-1).
- `output`, `storage`, `ingest`, `ordering`, `segmentation` tests all include
  genuine negative cases (traversal rejection, wrong-length tuple, broken rows,
  exception-no-sidecar, empty inputs). No muted/can't-fail tests found.
- `experiments/runner.py` subprocess isolation + timeout + skip/error rows is
  solid; smoke tests cover skip and error paths.

_End of audit._
