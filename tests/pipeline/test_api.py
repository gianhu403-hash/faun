"""API tests — run_pipeline is patched; the real chain is never invoked."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from faun import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with FAUN_JOBS_ROOT pointed at a temp dir."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    return TestClient(api.app)


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch run_pipeline so it writes a fake results.csv without the real chain."""
    calls = []

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        results = job_dir / "results.csv"
        results.write_text(
            "track,start_sec,duration_sec,species,probability\n"
            "A1,0.0,3.0,Turdus merula,0.91\n",
            encoding="utf-8",
        )
        calls.append((job_dir, source_path, lat, lon))
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)
    return calls


def test_full_cycle_post_poll_download(client, patched_pipeline):
    # POST /jobs (BackgroundTasks run synchronously inside TestClient request)
    resp = client.post(
        "/jobs", json={"source_path": "/data/A1", "lat": 58.1, "lon": 45.2}
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    # the background task ran -> run_pipeline was called once
    assert len(patched_pipeline) == 1
    assert patched_pipeline[0][1] == "/data/A1"

    # GET /jobs/{id} -> done
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["job_id"] == job_id
    assert body["status"] == "done"
    assert body["progress"] == 1.0

    # GET results.csv
    csv = client.get(f"/jobs/{job_id}/results.csv")
    assert csv.status_code == 200
    assert csv.headers["content-type"].startswith("text/csv")
    assert "Turdus merula" in csv.text


def test_post_accepts_url(client, patched_pipeline):
    resp = client.post("/jobs", json={"url": "https://example.com/rec.zip"})
    assert resp.status_code == 200
    assert patched_pipeline[0][1] == "https://example.com/rec.zip"


def test_get_unknown_job_404(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_results_404_when_missing(client, monkeypatch):
    # run_pipeline that does NOT write a csv -> results endpoint 404
    def no_csv(job_dir, source_path, lat=None, lon=None, classifier=None):
        return Path(job_dir) / "results.csv"

    monkeypatch.setattr(api, "run_pipeline", no_csv)
    resp = client.post("/jobs", json={"source_path": "/data/A1"})
    job_id = resp.json()["job_id"]
    csv = client.get(f"/jobs/{job_id}/results.csv")
    assert csv.status_code == 404


def test_validation_missing_source_and_url_422(client):
    resp = client.post("/jobs", json={"lat": 1.0})
    assert resp.status_code == 422


def test_error_status_when_pipeline_raises(client, monkeypatch):
    def boom(job_dir, source_path, lat=None, lon=None, classifier=None):
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(api, "run_pipeline", boom)
    resp = client.post("/jobs", json={"source_path": "/data/A1"})
    job_id = resp.json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "error"
    assert "ingest failed" in body["error"]


def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Faun" in resp.text


def test_store_is_faun_jobs(client, patched_pipeline, tmp_path):
    """The API persists through faun.jobs: the manifest it writes must be a valid
    faun.jobs Job (atomic store, single lifecycle), not a private api manifest."""
    from faun import jobs

    resp = client.post("/jobs", json={"source_path": "/data/A1"})
    job_id = resp.json()["job_id"]

    job = jobs.load_job(tmp_path / "jobs", job_id)
    assert job.status == "done"
    assert job.params["source_path"] == "/data/A1"
    assert job.params["progress"] == 1.0
    assert job.params["results_csv"] == "results.csv"
