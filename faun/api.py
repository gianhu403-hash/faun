"""Faun HTTP API (FastAPI) — POST /jobs, GET /jobs/{id}, results.csv.

Single integration point for the pipeline:

    run_pipeline(job_dir, source_path, lat, lon, classifier=None) -> Path

``run_pipeline`` lazily imports the Phase-2 modules (faun.ingest,
faun.segmentation, faun.classification, faun.output) inside its body, against
the frozen signatures in faun/INTERFACES.md, and runs the chain. Tests patch
``run_pipeline`` so the real (stub) chain is never exercised.

Job management is file-based (manifest.json per job dir under FAUN_JOBS_ROOT);
no module-level import of faun.jobs (it is a Phase-2 stub).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_CSV = "results.csv"
MANIFEST = "manifest.json"


def jobs_root() -> Path:
    """Resolve the jobs root (env FAUN_JOBS_ROOT, default ./jobs).

    Read at call time so tests can override the env var per-test.
    """
    root = Path(os.environ.get("FAUN_JOBS_ROOT", "./jobs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Manifest helpers (the minimal job store, file-based)
# ---------------------------------------------------------------------------


def _manifest_path(job_dir: Path) -> Path:
    return job_dir / MANIFEST


def read_manifest(job_dir: Path) -> Optional[dict]:
    path = _manifest_path(job_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(job_dir: Path, manifest: dict) -> None:
    # Atomic write (tmp + replace): GET /jobs/{id} polls concurrently with the
    # background task, a plain truncate+write would let it read torn JSON.
    path = _manifest_path(job_dir)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update_manifest(job_dir: Path, **changes) -> dict:
    manifest = read_manifest(job_dir) or {}
    manifest.update(changes)
    write_manifest(job_dir, manifest)
    return manifest


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

    Lazily imports the Phase-2 modules inside the body (they are stubs in this
    worktree; resolved at runtime against faun/INTERFACES.md). Returns the path
    to the written ``results.csv``.

    Tests patch this function — they do NOT call the real chain.
    """
    import soundfile as sf

    from faun import ingest, ordering, output, segmentation
    from faun.classification import StubAdapter

    if classifier is None:
        classifier = StubAdapter()

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    results_path = job_dir / RESULTS_CSV

    # ingest: scan(path) -> Manifest of AudioFileEntry; stable trap/time order
    manifest = ordering.sort_entries(ingest.scan(Path(source_path)))

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
    )

    extractor = segmentation.SegmentExtractor()
    with output.CsvWriter().open(results_path, meta=meta) as writer:
        for entry in manifest.entries:
            waveform, sr = sf.read(entry.path, dtype="float64", always_2d=False)
            for segment in extractor.extract(waveform, sr):
                for pred in classifier.classify(segment, sr):
                    writer.write_row(
                        {
                            "track": entry.trap_id,
                            "start_sec": segment.start_s,
                            "duration_sec": segment.duration_s,
                            "species": pred.species,
                            "probability": pred.probability,
                        }
                    )

    return results_path


# ---------------------------------------------------------------------------
# Background job execution
# ---------------------------------------------------------------------------


def _execute_job(
    job_id: str, source_path: str, lat: Optional[float], lon: Optional[float]
) -> None:
    job_dir = jobs_root() / job_id
    update_manifest(job_dir, status="running", progress=0.0)
    try:
        results = run_pipeline(job_dir, source_path, lat, lon)
        update_manifest(
            job_dir,
            status="done",
            progress=1.0,
            results_csv=str(Path(results).name),
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure in manifest
        update_manifest(job_dir, status="error", error=str(exc))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class JobRequest(BaseModel):
    source_path: Optional[str] = None
    url: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    @model_validator(mode="after")
    def _require_source(self) -> "JobRequest":
        if not self.source_path and not self.url:
            raise ValueError("either 'source_path' or 'url' is required")
        return self

    @property
    def source(self) -> str:
        return self.source_path or self.url  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Faun pipeline API")

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_html = _STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.post("/jobs")
def create_job(req: JobRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid.uuid4())
    job_dir = jobs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        job_dir,
        {
            "job_id": job_id,
            "status": "pending",
            "source_path": req.source,
            "lat": req.lat,
            "lon": req.lon,
            "progress": 0.0,
        },
    )
    background_tasks.add_task(_execute_job, job_id, req.source, req.lat, req.lon)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job_dir = jobs_root() / job_id
    manifest = read_manifest(job_dir)
    if manifest is None:
        raise HTTPException(status_code=404, detail="job not found")
    return manifest


@app.get("/jobs/{job_id}/results.csv")
def get_results(job_id: str) -> FileResponse:
    job_dir = jobs_root() / job_id
    results = job_dir / RESULTS_CSV
    if not results.exists():
        raise HTTPException(status_code=404, detail="results not available")
    return FileResponse(
        str(results), media_type="text/csv", filename=f"{job_id}_results.csv"
    )
