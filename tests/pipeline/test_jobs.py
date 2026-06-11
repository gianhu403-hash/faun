"""Tests for faun.jobs — job lifecycle, workdir isolation, manifest atomicity."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from faun.jobs import VALID_STATUSES, Job, create_job, load_job, update_status


class TestCreateJob:
    def test_creates_isolated_workdir_with_manifest(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {"source_path": "/data/A1", "lat": 57.4})
        assert isinstance(job.job_id, uuid.UUID)
        assert job.workdir == tmp_path / str(job.job_id)
        assert job.workdir.is_dir()
        assert job.status == "pending"
        assert job.manifest_path.is_file()

        data = json.loads(job.manifest_path.read_text())
        assert data["job_id"] == str(job.job_id)
        assert data["status"] == "pending"
        assert data["params"] == {"source_path": "/data/A1", "lat": 57.4}
        assert data["created_at"] == data["updated_at"] != ""

    def test_two_jobs_have_distinct_workdirs(self, tmp_path: Path) -> None:
        job_a = create_job(tmp_path, {})
        job_b = create_job(tmp_path, {})
        assert job_a.job_id != job_b.job_id
        assert job_a.workdir != job_b.workdir
        assert job_a.workdir.is_dir() and job_b.workdir.is_dir()

    def test_params_are_copied_not_shared(self, tmp_path: Path) -> None:
        params = {"x": 1}
        job = create_job(tmp_path, params)
        params["x"] = 999
        assert job.params == {"x": 1}

    def test_results_path_inside_workdir(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        assert job.results_path == job.workdir / "results.csv"

    def test_no_tmp_files_left_behind(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        update_status(job, "running")
        leftovers = [p for p in job.workdir.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


class TestStatus:
    def test_full_lifecycle_persists(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        for status in ("running", "done"):
            update_status(job, status)
            assert job.status == status
            reloaded = load_job(tmp_path, job.job_id)
            assert reloaded.status == status

    def test_error_status(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        update_status(job, "error")
        assert load_job(tmp_path, job.job_id).status == "error"

    def test_invalid_status_raises_and_keeps_manifest(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        with pytest.raises(ValueError):
            update_status(job, "exploded")
        assert json.loads(job.manifest_path.read_text())["status"] == "pending"

    def test_updated_at_advances(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        created = job.created_at
        update_status(job, "running")
        assert job.updated_at >= created
        assert job.created_at == created


class TestLoadJob:
    def test_roundtrip(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {"gain": 12, "channel": "mono"})
        loaded = load_job(tmp_path, job.job_id)
        assert isinstance(loaded, Job)
        assert loaded.job_id == job.job_id
        assert loaded.workdir == job.workdir
        assert loaded.params == {"gain": 12, "channel": "mono"}
        assert loaded.created_at == job.created_at

    def test_accepts_string_job_id(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        loaded = load_job(tmp_path, str(job.job_id))
        assert loaded.job_id == job.job_id

    def test_missing_job_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_job(tmp_path, uuid.uuid4())

    def test_corrupt_status_raises_value_error(self, tmp_path: Path) -> None:
        job = create_job(tmp_path, {})
        data = json.loads(job.manifest_path.read_text())
        data["status"] = "weird"
        job.manifest_path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_job(tmp_path, job.job_id)


def test_valid_statuses_contract() -> None:
    assert VALID_STATUSES == {"pending", "running", "done", "error"}
