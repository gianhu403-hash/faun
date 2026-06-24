"""Faun HTTP API (FastAPI) — POST /jobs, GET /jobs/{id}, results.csv.

Single integration point for the pipeline:

    run_pipeline(job_dir, source_path, lat, lon, classifier=None) -> Path

``run_pipeline`` lazily imports the Phase-2 modules (faun.ingest,
faun.segmentation, faun.classification, faun.output) inside its body, against
the frozen signatures in faun/INTERFACES.md, and runs the chain. Tests patch
``run_pipeline`` so the real (stub) chain is never exercised.

Job management is delegated to faun.jobs (atomic manifest write via tmp+rename,
pending -> running -> done|error lifecycle). The API keeps its public JSON shape
by flattening the job's ``params`` (source_path, lat, lon, progress, results_csv,
error) alongside the top-level ``job_id``/``status``.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hmac
import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from faun import jobs
from faun.obs import setup_logging, with_job_context
from faun.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_CSV = "results.csv"
DETECTIONS_JSONL = "detections.jsonl"
SPECIES_PRESENCE_JSON = "species_presence.json"
PROB_SMOOTHED_JSON = "prob_smoothed.json"
SEGMENTS_DIR = "segments"
DETECTIONS_LOCK = ".detections.lock"

#: detection_id is a uuid4 hex (optionally dashed); reject anything else to
#: prevent path traversal on the clip-download route.
_DETECTION_ID_RE = re.compile(r"^[0-9a-fA-F-]+$")


def jobs_root() -> Path:
    """Resolve the jobs root (FAUN_JOBS_ROOT, default ./jobs) via faun.settings.

    Routed through the cached ``get_settings()``; tests clear that cache
    (``get_settings.cache_clear()`` — done in the conftest autouse fixture) so a
    per-test FAUN_JOBS_ROOT override is still picked up at call time.
    """
    root = get_settings().jobs_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_view(job: jobs.Job) -> dict:
    """Flatten a Job into the public API response shape.

    Top-level ``job_id`` + ``status`` (from the Job) merged with the API-level
    params (source_path, lat, lon, progress, results_csv, error). This keeps the
    external JSON contract unchanged while the store lives in faun.jobs.
    """
    return {
        "job_id": str(job.job_id),
        "status": job.status,
        **job.params,
    }


# ---------------------------------------------------------------------------
# Classifier selection (cluster-only; CI/local stays Stub)
# ---------------------------------------------------------------------------


def _build_base_classifier():
    """Build the bare classifier from FAUN_CLASSIFIER (default ``stub``).

    Heavy adapters are imported lazily via faun.classification (PEP-562) only
    when chosen, so importing this module never pulls TensorFlow.
    """
    choice = get_settings().classifier
    if choice in ("", "stub"):
        from faun.classification import StubAdapter

        return StubAdapter()
    if choice == "perch":
        from faun.classification import PerchAdapter

        return PerchAdapter()
    if choice == "perch-v2":
        from faun.classification import Perch2Adapter

        return Perch2Adapter()
    if choice == "perch-probe":
        from faun.classification import PerchProbeAdapter

        return PerchProbeAdapter()
    if choice == "birdnet":
        from faun.classification import BirdNETAdapter

        return BirdNETAdapter()
    if choice == "yamnet":
        from faun.classification import YAMNetAdapter

        return YAMNetAdapter()
    raise ValueError(
        f"unknown FAUN_CLASSIFIER={choice!r}; expected one of "
        "stub, perch, perch-v2, perch-probe, birdnet, yamnet"
    )


def _allowlist_vocab_provider(clf):
    """Return a callable yielding ``clf``'s label vocabulary, or ``None``.

    Used by :class:`~faun.classification.MaskedClassifier` to validate allow-list
    coverage. Only the scientific-name adapters expose a rich vocabulary; for the
    rest the mask cannot verify coverage and stays a no-op (fail-open). Reads the
    adapter's own (lazy) label source — resolved only after the first inference,
    which is exactly when the coverage gate runs.
    """
    name = clf.__class__.__name__
    if name == "Perch2Adapter":
        return clf._load_labels
    if name == "PerchProbeAdapter":

        def _probe_vocab():
            probe = clf._load_probe()
            classes = getattr(probe, "classes_", None) if probe is not None else None
            return [str(c) for c in classes] if classes is not None else None

        return _probe_vocab
    return None


def _build_classifier():
    """Build the serving classifier, applying the regional allow-list if set.

    When ``FAUN_SPECIES_ALLOWLIST`` is configured (and yields a non-empty
    checklist), the base classifier is wrapped in a
    :class:`~faun.classification.MaskedClassifier` so the argmax is restricted to
    regional species (ADR-0004). UNSET -> the bare classifier is returned and the
    output is byte-for-byte unchanged (the default, prod-safe path).
    """
    base = _build_base_classifier()
    spec = get_settings().species_allowlist
    if not spec:
        return base
    from faun.classification import MaskedClassifier, load_allowlist

    allow = load_allowlist(spec)
    if not allow:
        # load_allowlist already logged the misconfiguration (missing / empty /
        # unreadable); leave the classifier unmasked rather than wrapping a no-op.
        return base
    return MaskedClassifier(base, allow, vocab_provider=_allowlist_vocab_provider(base))


def _classifier_source(classifier) -> str:
    """Map a classifier instance to its detections ``source`` tag.

    Avoids importing the heavy adapter classes: matches on class name so the
    module stays TensorFlow-free at import time.
    """
    from faun.detections import (
        SOURCE_BIRDNET,
        SOURCE_PERCH,
        SOURCE_PERCH_V2,
        SOURCE_PERCH_V2_PROBE,
        SOURCE_STUB,
        SOURCE_YAMNET_PROBE,
    )

    # Unwrap a MaskedClassifier (ADR-0004): the source tag must reflect the real
    # backbone (e.g. "model:perch-v2"), not the wrapper, so detections.jsonl
    # provenance is unchanged when the regional mask is enabled. Loop in case a
    # future change ever stacks wrappers.
    while hasattr(classifier, "inner"):
        classifier = classifier.inner
    name = classifier.__class__.__name__
    mapping = {
        "StubAdapter": SOURCE_STUB,
        "PerchAdapter": SOURCE_PERCH,
        "Perch2Adapter": SOURCE_PERCH_V2,
        "PerchProbeAdapter": SOURCE_PERCH_V2_PROBE,
        "BirdNETAdapter": SOURCE_BIRDNET,
        "YAMNetAdapter": SOURCE_YAMNET_PROBE,
    }
    return mapping.get(name, f"model:{name.lower()}")


# ---------------------------------------------------------------------------
# The single pipeline integration point
# ---------------------------------------------------------------------------


def run_pipeline(
    job_dir: Path,
    source_path: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    classifier=None,
) -> Path:
    """Run the full ingest -> segment -> classify -> output chain.

    Lazily imports the Phase-2 modules inside the body (resolved at runtime
    against faun/INTERFACES.md). Writes ``results.csv`` + ``results_meta.json``
    (unchanged) and, additionally, persists per-event ``detections.jsonl`` plus
    real ``segments/<detection_id>.wav`` clips cut from the ORIGINAL waveform at
    the ORIGINAL samplerate. Returns the path to the written ``results.csv``.

    Tests patch this function — they do NOT call the real chain.
    """
    import shutil

    import soundfile as sf

    from faun import ingest, ordering, output, pipeline as pl
    from faun.detections import Label, write_detections, STATUS_PSEUDO
    from faun.sources import resolve_source, source_provenance

    if classifier is None:
        classifier = _build_classifier()
    source = _classifier_source(classifier)

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    results_path = job_dir / RESULTS_CSV
    segments_dir = job_dir / SEGMENTS_DIR

    try:
        # Resolve the source to a local scannable dir. NEVER Path() a URL: a local
        # path passes through unchanged; an http(s) zip / Yandex.Disk share is
        # downloaded + safely extracted under job_dir/_source/ (the P0 fix). A
        # SourceError here propagates to _execute_job -> job status="error".
        scan_dir = resolve_source(source_path, job_dir)
        # ingest: scan(path) -> Manifest of AudioFileEntry; stable trap/time order
        manifest = ordering.sort_entries(ingest.scan(scan_dir))

        # Sidecar provenance: only claim a trap_id/coords when the batch is
        # single-trap; a mixed batch gets "multi" and no coordinates (honest
        # rather than first-entry's identity stamped onto everyone's rows).
        trap_ids = {e.trap_id for e in manifest.entries}
        first = manifest.entries[0] if manifest.entries else None
        single = len(trap_ids) == 1 and first is not None
        meta = output.TrapMeta(
            trap_id=(first.trap_id if single else ("multi" if trap_ids else "")),
            lat=lat if lat is not None else (first.lat if single else None),
            lon=lon if lon is not None else (first.lon if single else None),
            files=[e.path.name for e in manifest.entries],
            extra={"source_provenance": source_provenance(source_path)},
        )

        # Read at float64 for clip fidelity; the classifier gets a 16 kHz mono
        # COPY (faun.pipeline.to_classifier_input) while the on-disk clip keeps
        # the ORIGINAL sr + channels.
        def _read(path):
            return sf.read(path, dtype="float64", always_2d=False)

        def _build_labels(clip_16k):
            return [
                Label.from_prediction(pred, source=source, status=STATUS_PSEUDO)
                for pred in classifier.classify(clip_16k, pl.CLASSIFY_SR)
            ]

        # run_batch yields one (Detection, clip) at a time, so we write each clip
        # + its CSV rows and keep only ONE clip live (the real ingest can be tens
        # of GB). Detections (metadata only) accumulate for detections.jsonl.
        detections = []
        with output.CsvWriter().open(results_path, meta=meta) as writer:
            for res in pl.run_batch(
                manifest.entries, read_waveform=_read, build_labels=_build_labels
            ):
                det = res.detection
                # Write the clip from the ORIGINAL waveform at the ORIGINAL sr;
                # det.segment_path's basename matches the file we write here.
                segments_dir.mkdir(parents=True, exist_ok=True)
                sf.write(job_dir / det.segment_path, res.clip, res.sr)
                detections.append(det)
                for lbl in det.labels:
                    writer.write_row(
                        {
                            "track": det.trap_id,
                            "start_sec": det.segment.start_s,
                            "duration_sec": det.segment.duration_s,
                            "species": lbl.species,
                            "probability": lbl.probability,
                        }
                    )

        # Always write detections.jsonl — empty (zero lines) for silent input.
        write_detections(job_dir / DETECTIONS_JSONL, detections)
        # Biologist deliverable: per-trap, per-day species presence aggregate
        # (FR-005, ADR-0004). Output-only — derived from the detections we just
        # built, after the streaming loop; the CSV / streaming path is untouched.
        output.write_species_presence(job_dir / SPECIES_PRESENCE_JSON, detections)
        # Per-recording probability smoothing sidecar (FR-004, ADR-0006), OFF by
        # default: only written when FAUN_PROB_SMOOTHING is set, so the default
        # job directory (and the golden diff) is unchanged. Output-only, same
        # post-loop position as species_presence; raw probability untouched.
        if get_settings().prob_smoothing:
            output.write_prob_smoothed(job_dir / PROB_SMOOTHED_JSON, detections)

        return results_path
    finally:
        # B1 disk-leak: drop the downloaded/extracted REMOTE source tree on exit
        # (success OR error) — its files were only needed during segmentation
        # above. resolve_source creates job_dir/_source ONLY for a remote source;
        # a local pass-through source is Path(src) OUTSIDE job_dir, never touched.
        shutil.rmtree(job_dir / "_source", ignore_errors=True)


# ---------------------------------------------------------------------------
# Background job execution
# ---------------------------------------------------------------------------


def _execute_job(
    job_id: str, source_path: str, lat: Optional[float], lon: Optional[float]
) -> None:
    with with_job_context(job_id):
        root = jobs_root()
        job = jobs.load_job(root, job_id)
        job.params["progress"] = 0.0
        jobs.update_status(job, "running")
        try:
            results = run_pipeline(job.workdir, source_path, lat, lon)
            job.params["progress"] = 1.0
            job.params["results_csv"] = str(Path(results).name)
            jobs.update_status(job, "done")
        except Exception as exc:  # noqa: BLE001 — surface any failure in manifest
            logger.exception("job failed")  # job_id is a structured log field now
            job.params["error"] = str(exc)
            # SourceError carries a stable .kind taxonomy the UI maps to RU labels.
            kind = getattr(exc, "kind", None)
            if kind:
                job.params["error_kind"] = kind
            jobs.update_status(job, "error")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class JobRequest(BaseModel):
    source_path: Optional[str] = None
    url: Optional[str] = None
    # WGS84 bounds; the constraints also reject NaN/Inf (a bound check on a
    # non-finite float fails) -> 422 instead of poisoning the sidecar coords.
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def _require_source(self) -> "JobRequest":
        if not self.source_path and not self.url:
            raise ValueError("either 'source_path' or 'url' is required")
        return self

    @property
    def source(self) -> str:
        return self.source_path or self.url  # type: ignore[return-value]


class LabelRequest(BaseModel):
    species: str = Field(min_length=1, max_length=200)
    source: str = Field(default="operator:ranger", max_length=100)
    status: Optional[str] = None  # default resolved to STATUS_CORRECTED at use


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Faun pipeline API")

# Structured logging configured once at import time (idempotent); JSON by default.
setup_logging(json=get_settings().log_json)


# ---------------------------------------------------------------------------
# HTTP Basic Auth (env-gated; default-OPEN so CI / local / tests are unchanged)
# ---------------------------------------------------------------------------

#: Paths served WITHOUT auth even when Basic Auth is enabled. /healthz must stay
#: open for the container HEALTHCHECK / external liveness probe.
_AUTH_OPEN_PATHS = frozenset({"/healthz"})


@app.middleware("http")
async def _basic_auth_mw(request: Request, call_next):
    """Site-wide HTTP Basic Auth gate (the simple browser login the team shares).

    Enabled ONLY when BOTH ``FAUN_BASIC_USER`` and ``FAUN_BASIC_PASS`` are set
    (via faun.settings); otherwise every request passes through unchanged
    (default-open — CI, local dev and the existing test-suite see no auth). When
    enabled, every path EXCEPT ``/healthz`` requires valid credentials: the SPA
    pages, the POST routes and the ``/static`` mount alike, so a browser shows
    the native login dialog on first visit. Credentials are checked with
    ``hmac.compare_digest`` (constant-time); a miss returns 401 +
    ``WWW-Authenticate: Basic`` so the browser re-prompts.
    """
    settings = get_settings()
    user, password = settings.basic_user, settings.basic_pass
    if not user or not password:
        return await call_next(request)  # auth disabled -> open
    if request.url.path in _AUTH_OPEN_PATHS:
        return await call_next(request)

    header = request.headers.get("Authorization", "")
    if header[:6].lower() == "basic ":
        try:
            raw = base64.b64decode(header[6:], validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            raw = ""
        req_user, sep, req_pass = raw.partition(":")
        # Evaluate BOTH halves (no boolean short-circuit) for constant-time-ish
        # comparison; encode to bytes so non-ASCII creds don't raise.
        ok_user = hmac.compare_digest(req_user.encode("utf-8"), user.encode("utf-8"))
        ok_pass = hmac.compare_digest(
            req_pass.encode("utf-8"), password.encode("utf-8")
        )
        if sep and ok_user and ok_pass:
            return await call_next(request)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Faun", charset="UTF-8"'},
    )


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_html = _STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    page = _STATIC_DIR / "dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/review", response_class=HTMLResponse)
def review() -> HTMLResponse:
    page = _STATIC_DIR / "review.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="review.html not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz() -> dict:
    """Лёгкий liveness/readiness без тяжёлых импортов (handler в faun.health)."""
    from faun.health import health

    return health()


@app.post("/jobs")
def create_job(req: JobRequest, background_tasks: BackgroundTasks) -> dict:
    job = jobs.create_job(
        jobs_root(),
        params={
            "source_path": req.source,
            "lat": req.lat,
            "lon": req.lon,
            "progress": 0.0,
        },
    )
    job_id = str(job.job_id)
    background_tasks.add_task(_execute_job, job_id, req.source, req.lat, req.lon)
    return {"job_id": job_id}


@app.get("/jobs")
def list_jobs() -> list[dict]:
    """List every job (for the dashboard: trap positions + queue state).

    Scans jobs_root for ``<id>/manifest.json``, loads each via faun.jobs and
    flattens it through ``_job_view``. Corrupt/partial dirs are skipped with a
    warning rather than failing the whole listing. Sorted by ``created_at``.
    """
    root = jobs_root()
    views: list[dict] = []
    for manifest in root.glob("*/manifest.json"):
        job_id = manifest.parent.name
        try:
            job = jobs.load_job(root, job_id)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping unreadable job %s: %s", job_id, exc)
            continue
        view = _job_view(job)
        view["created_at"] = job.created_at
        views.append(view)
    views.sort(key=lambda v: v.get("created_at", ""))
    return views


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        job = jobs.load_job(jobs_root(), job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="job not found")
    return _job_view(job)


@app.get("/jobs/{job_id}/results.csv")
def get_results(job_id: str) -> FileResponse:
    job_dir = jobs_root() / job_id
    results = job_dir / RESULTS_CSV
    if not results.exists():
        raise HTTPException(status_code=404, detail="results not available")
    return FileResponse(
        str(results), media_type="text/csv", filename=f"{job_id}_results.csv"
    )


def _job_dir_or_404(job_id: str) -> Path:
    """Resolve job_dir, 404 if the job/manifest is missing."""
    try:
        job = jobs.load_job(jobs_root(), job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="job not found")
    return job.workdir


@app.get("/jobs/{job_id}/detections")
def get_detections(job_id: str) -> list[dict]:
    """Return the job's per-event detections (empty list if not yet written).

    404 only when the job/manifest is missing; a job that ran before this
    feature (no detections.jsonl) returns ``[]``, not a 500. Each detection
    carries a ``localization`` key (null for now — co-detection is deferred).
    """
    from faun.detections import read_detections

    job_dir = _job_dir_or_404(job_id)
    jsonl = job_dir / DETECTIONS_JSONL
    if not jsonl.exists():
        return []
    try:
        dets = read_detections(jsonl)
    except Exception:
        logger.exception("failed to read detections for job %s", job_id)
        raise HTTPException(status_code=500, detail="detections unreadable")
    out: list[dict] = []
    for det in dets:
        view = det.to_dict()
        view["localization"] = None
        out.append(view)
    return out


@app.get("/jobs/{job_id}/segments/{detection_id}.wav")
def get_segment(job_id: str, detection_id: str) -> FileResponse:
    """Serve a detection's real audio clip; reject non-hex ids (traversal)."""
    if not _DETECTION_ID_RE.match(detection_id):
        raise HTTPException(status_code=404, detail="segment not found")
    job_dir = _job_dir_or_404(job_id)
    clip = job_dir / SEGMENTS_DIR / f"{detection_id}.wav"
    if not clip.is_file():
        raise HTTPException(status_code=404, detail="segment not found")
    return FileResponse(str(clip), media_type="audio/wav")


@app.get("/jobs/{job_id}/segments/{detection_id}.png")
def get_segment_spectrogram(job_id: str, detection_id: str) -> FileResponse:
    """Serve a detection clip's spectrogram PNG (FR-008), rendered lazily.

    Mirrors :func:`get_segment` — same hex-id traversal guard — and renders the
    PNG from the existing ``segments/<id>.wav`` on first request, caching it next
    to the clip so subsequent loads (and the review grid's lazy <img>s) are a
    plain file read. A render failure is a 500, never a crash of the route.
    """
    if not _DETECTION_ID_RE.match(detection_id):
        raise HTTPException(status_code=404, detail="segment not found")
    job_dir = _job_dir_or_404(job_id)
    clip = job_dir / SEGMENTS_DIR / f"{detection_id}.wav"
    if not clip.is_file():
        raise HTTPException(status_code=404, detail="segment not found")
    png = job_dir / SEGMENTS_DIR / f"{detection_id}.png"
    if not png.is_file():
        import os
        import tempfile

        import soundfile as sf

        from faun.spectrogram import save_spectrogram_png

        try:
            wav, sr = sf.read(str(clip), dtype="float64", always_2d=False)
            # Render to a UNIQUE temp file (suffix .png so matplotlib picks the
            # PNG format) then atomically rename, so a concurrent double-miss on
            # the same clip can never serve a half-written PNG (mirrors
            # faun.detections.write_detections' tmp+replace idiom).
            fd, tmp_name = tempfile.mkstemp(
                dir=str(png.parent), prefix=f".{detection_id}.", suffix=".png"
            )
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                save_spectrogram_png(wav, int(sr), tmp)
                tmp.replace(png)
            finally:
                tmp.unlink(missing_ok=True)  # no-op if replace already moved it
        except Exception as exc:  # noqa: BLE001 — surface render failure as 500
            logger.exception("spectrogram render failed for %s", detection_id)
            raise HTTPException(
                status_code=500, detail="spectrogram render failed"
            ) from exc
    return FileResponse(str(png), media_type="image/png")


@app.post("/jobs/{job_id}/detections/{detection_id}/label")
def add_label(job_id: str, detection_id: str, req: LabelRequest) -> dict:
    """Append a human label to a detection (concurrency-safe read-modify-write).

    The whole read+modify+write of detections.jsonl is guarded by an
    ``fcntl.flock`` on ``<job_dir>/.detections.lock`` so two concurrent POSTs
    cannot lose an update. 404 if the job or the detection_id is unknown.
    """
    from faun.detections import (
        Label,
        read_detections,
        write_detections,
        STATUS_CORRECTED,
    )

    if not _DETECTION_ID_RE.match(detection_id):
        raise HTTPException(status_code=404, detail="detection not found")
    job_dir = _job_dir_or_404(job_id)
    jsonl = job_dir / DETECTIONS_JSONL
    lock_path = job_dir / DETECTIONS_LOCK
    status = req.status or STATUS_CORRECTED

    try:
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                if not jsonl.exists():
                    raise HTTPException(status_code=404, detail="detection not found")
                dets = read_detections(jsonl)
                target = next((d for d in dets if d.detection_id == detection_id), None)
                if target is None:
                    raise HTTPException(status_code=404, detail="detection not found")
                target.labels.append(
                    Label.now(
                        species=req.species,
                        probability=None,
                        source=req.source,
                        status=status,
                    )
                )
                write_detections(jsonl, dets)
                return target.to_dict()
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
    except HTTPException:
        raise
    except Exception:
        logger.exception("failed to label detection %s in job %s", detection_id, job_id)
        raise HTTPException(status_code=500, detail="label failed")
