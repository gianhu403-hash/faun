"""Jobs: per-job_id workdir isolation, status, manifest, results (Phase 2).

Contract (faun/INTERFACES.md, frozen):
    Job(job_id: uuid, workdir=jobs_root/<job_id>/, status, manifest.json,
    results.csv); batch isolation = namespace per job_id, no shared temp paths.

Lifecycle: pending -> running -> done | error.
manifest.json is written atomically (tmp + rename).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["Job", "VALID_STATUSES", "create_job", "update_status", "load_job"]

VALID_STATUSES = frozenset({"pending", "running", "done", "error"})

MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.csv"


@dataclass
class Job:
    """A processing job with an isolated workdir under jobs_root/<job_id>/."""

    job_id: uuid.UUID
    workdir: Path
    status: str
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def manifest_path(self) -> Path:
        return self.workdir / MANIFEST_NAME

    @property
    def results_path(self) -> Path:
        return self.workdir / RESULTS_NAME


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(job: Job) -> None:
    """Atomically persist the job manifest (tmp + rename, same directory)."""
    payload = {
        "job_id": str(job.job_id),
        "status": job.status,
        "params": job.params,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    tmp_path = job.workdir / f".{MANIFEST_NAME}.tmp"
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp_path, job.manifest_path)


def create_job(jobs_root: Path, params: dict[str, Any]) -> Job:
    """Create a new job: isolated workdir jobs_root/<job_id>/ + manifest.json."""
    job_id = uuid.uuid4()
    workdir = Path(jobs_root) / str(job_id)
    workdir.mkdir(parents=True, exist_ok=False)
    now = _utcnow()
    job = Job(
        job_id=job_id,
        workdir=workdir,
        status="pending",
        params=dict(params),
        created_at=now,
        updated_at=now,
    )
    _write_manifest(job)
    return job


def update_status(job: Job, status: str) -> Job:
    """Transition the job to a new status and persist the manifest."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"invalid status {status!r}, expected one of {sorted(VALID_STATUSES)}"
        )
    job.status = status
    job.updated_at = _utcnow()
    _write_manifest(job)
    return job


def load_job(jobs_root: Path, job_id: uuid.UUID | str) -> Job:
    """Reload a job from jobs_root/<job_id>/manifest.json."""
    job_id = uuid.UUID(str(job_id))
    workdir = Path(jobs_root) / str(job_id)
    manifest_path = workdir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest for job {job_id} at {manifest_path}")
    data = json.loads(manifest_path.read_text())
    status = data.get("status", "")
    if status not in VALID_STATUSES:
        raise ValueError(f"corrupt manifest {manifest_path}: invalid status {status!r}")
    return Job(
        job_id=job_id,
        workdir=workdir,
        status=status,
        params=data.get("params", {}),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
