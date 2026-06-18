# ADR 0001 — Source-resolution layer in front of ingest

- Status: Accepted
- Date: 2026-06-19
- Deciders: Faun v2 pipeline team
- Supersedes: —
- Branch of record: `feat/perch2-finetune-ingest`

## Context

The Faun v2 product accepts a batch of acoustic-trap recordings either as a
local **folder** or as a **URL** (`POST /jobs {source_path|url, ...}`). The
folder case is handled by the frozen ingest contract:

```python
# faun/ingest: scan(path: Path) -> Manifest   (FROZEN — faun/INTERFACES.md)
```

`ingest.scan` only knows how to walk a local directory of one-folder-per-trap
recordings (`info.txt` + audio). It has no notion of the network. Yet today the
API already advertises a `url` field on `JobRequest`, and operators will paste
Yandex.Disk public links and plain archive URLs. The pipeline therefore needs a
step that turns *any* accepted source into a **local directory** that
`ingest.scan` can consume — without changing `scan`'s frozen signature.

Three concrete problems pushed this from "nice to have" to "do it before
ingest":

1. **P0 bug — `url` reaches `ingest.scan` as a path.** In `faun/api.py`,
   `run_pipeline(... source_path ...)` calls `ingest.scan(Path(source_path))`
   directly (api.py:174). When a job is created from a `url`
   (`JobRequest.source` returns `self.source_path or self.url`, api.py:282), the
   URL string is wrapped in `Path(...)` and handed to a local-filesystem
   scanner. The job fails deep inside ingest with an opaque "no such directory"
   instead of being resolved or rejected up front. The URL path is effectively
   unimplemented and silently broken.

2. **SSRF / zip-bomb exposure.** Once we *do* fetch URLs, an unhardened
   downloader is a server-side-request-forgery and decompression-bomb vector:
   redirects to internal addresses, unbounded download size, archives that
   expand to terabytes, or tens of thousands of entries. These limits must live
   somewhere typed and central, not as magic numbers sprinkled across a fetch
   helper.

3. **Blue-green deploy / honest backend.** `faun.antopkin.ru/` now routes to the
   v2 container (`100.64.0.1:8010`) running with `FAUN_CLASSIFIER=stub` — a
   demo-facing, blue-green swap from the frozen v1 demo. A job that 500s on a
   pasted Yandex.Disk link in front of the customer is unacceptable; the source
   step must fail *cleanly and early* (clear 4xx) when it cannot resolve a
   source, and the deploy must configure its limits through one documented set
   of env vars rather than code edits.

Configuration today is read ad-hoc: `FAUN_JOBS_ROOT` and `FAUN_CLASSIFIER` are
parsed inline in `faun/api.py` (api.py:59, api.py:89), `faun/health.py`
re-implements the jobs-root resolution (health.py:36), and each model adapter
reads its own `os.environ` (`PERCH_MODEL_PATH`, `YAMNET_PROBE_PATH`). Adding a
fetch layer with its own limits would multiply this drift.

## Decision

Introduce a **source-resolution layer** (`faun/sources`) that runs **before**
ingest and is responsible for normalizing any accepted source into a local
directory:

```
POST /jobs {source_path|url}
        │
        ▼
faun.sources.resolve(source, *, settings) -> Path   # local dir
        │
        ▼
ingest.scan(local_dir)   # FROZEN signature, unchanged
        │
        ▼
segmentation -> classification -> output
```

Design rules:

1. **Ingest stays frozen.** `faun.sources.resolve` returns a `Path` to a local
   directory; `ingest.scan` is called with that path and is otherwise untouched.
   A local `source_path` resolves trivially to itself (no copy); only remote
   sources are fetched/unpacked into the job workdir.

2. **Hardening lives in the resolver, governed by typed Settings.** The resolver
   enforces, from `faun.settings.Settings`:
   - `timeout_s` — per-request network timeout (default 30 s).
   - `max_bytes` — total download size cap (default 2 GiB).
   - `max_uncompressed_bytes` — cumulative inflated archive size cap, the
     zip-bomb guard (default 4 GiB).
   - `max_entries` — archive member count cap (default 10 000).
   - `max_redirects` — redirect-follow cap, the SSRF guard (default 5), with
     scheme/host validation on each hop.

3. **Centralized, typed configuration.** All `FAUN_*` knobs — including the new
   `FAUN_SOURCE_*` limits and the model-path variables — are read once into the
   frozen `faun.settings.Settings` dataclass via `Settings.from_env`, exposed
   through the cached `get_settings()`. Numeric vars are parsed *defensively*: a
   malformed or non-positive value falls back to the documented default and logs
   a warning rather than crashing boot. `faun.api` and `faun.health` consume
   `get_settings()` instead of their own `os.environ` calls.

4. **Structured, job-scoped logging.** `faun.obs` provides `setup_logging(json=)`
   (idempotent JSON logs, stdlib only) and `with_job_context(job_id)`, replacing
   bare `logger.exception("job %s failed", job_id)` so a resolution or pipeline
   failure carries `job_id` as a structured field — essential when triaging the
   blue-green demo from logs alone.

This ADR records the *layer and its configuration/observability substrate*. The
concrete fetchers (HTTP archive, Yandex.Disk public-link API) are implemented as
a sibling vector against this contract; `faun/settings.py` and `faun/obs.py`
ship now as their foundation.

## Consequences

Positive:

- The P0 `url`-as-path bug gets a real home to be fixed: `resolve` either
  produces a local dir or raises a typed error the API maps to a clean 4xx,
  instead of failing opaquely inside ingest.
- SSRF and zip-bomb limits are explicit, typed, defaulted, and overridable per
  deploy via documented `FAUN_SOURCE_*` env vars — no code edits to retune.
- One configuration source of truth removes the `FAUN_JOBS_ROOT` /
  `FAUN_CLASSIFIER` / model-path duplication across `api`, `health`, and the
  adapters.
- Structured logs with `job_id` make the blue-green demo debuggable from log
  aggregation without code changes.
- The frozen ingest contract is preserved; existing local-folder jobs are
  unaffected (local paths resolve to themselves).

Negative / costs:

- One more module on the request path; local-folder jobs gain a (cheap)
  pass-through resolution step.
- Defensive env parsing means a typo'd limit is *silently* defaulted (with a
  warning). Operators must read logs to notice a mis-set limit rather than
  getting a hard failure. This is the deliberate trade: availability over
  fail-fast for a customer-facing demo.
- The model-path env vars are now *also* surfaced via Settings while the
  adapters still read `os.environ` directly; until the adapters are migrated
  there are two readers of the same variable (consistent values, but a follow-up
  is needed to route adapters through Settings).

## References

- `faun/settings.py` — `Settings` (frozen) + `get_settings()` (cached,
  invalidatable).
- `faun/obs.py` — `setup_logging`, `with_job_context`.
- `faun/INTERFACES.md` — frozen `ingest.scan(path) -> Manifest` (unchanged).
- `faun/api.py:174` — current direct `ingest.scan(Path(source_path))` call (the
  P0 site); `faun/api.py:282` — `JobRequest.source` returning the URL string.
- Blue-green deploy: `faun.antopkin.ru/` → v2 container `100.64.0.1:8010`
  (`FAUN_CLASSIFIER=stub`), v1 demo moved to `faun.antopkin.ru/v1/`.
