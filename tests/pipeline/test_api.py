"""API tests — run_pipeline is patched; the real chain is never invoked."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from faun import api, jobs
from faun.detections import (
    Detection,
    Label,
    read_detections,
    write_detections,
    SOURCE_STUB,
    STATUS_PSEUDO,
)
from faun.segmentation import Segment


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


def test_healthz_ok(client):
    """/healthz is wired to faun.health.health() and returns a 200 readiness body."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "faun-api"
    assert "version" in body


# ---------------------------------------------------------------------------
# Phase-3 frontend render backstop (gate 6): the three windows + shared assets
# ---------------------------------------------------------------------------


def test_index_references_shared_assets(client):
    """index.html must link the shared stylesheet + helpers (no inline-only)."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/static/app.js" in resp.text
    assert "/static/styles.css" in resp.text


def test_dashboard_served(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    # Map container + Leaflet dependency mark this as the dashboard window.
    assert 'id="map"' in resp.text
    assert "Leaflet" in resp.text or "leaflet" in resp.text
    assert "/static/app.js" in resp.text
    assert "/static/styles.css" in resp.text


def test_review_served(client):
    resp = client.get("/review")
    assert resp.status_code == 200
    # Real audio player + review wording mark this as the relabel window.
    assert "<audio" in resp.text
    assert "переразметка" in resp.text.lower()
    assert "/static/app.js" in resp.text
    assert "/static/styles.css" in resp.text


def test_static_assets_served(client):
    """StaticFiles mount serves the shared JS + CSS bundles."""
    js = client.get("/static/app.js")
    assert js.status_code == 200
    css = client.get("/static/styles.css")
    assert css.status_code == 200


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


# ---------------------------------------------------------------------------
# GET /jobs — listing
# ---------------------------------------------------------------------------


def test_list_jobs_returns_created_jobs(client, patched_pipeline):
    """Both created jobs appear in GET /jobs (dashboard listing)."""
    ids = set()
    for src in ("/data/A1", "/data/A2"):
        resp = client.post("/jobs", json={"source_path": src, "lat": 58.0, "lon": 45.0})
        assert resp.status_code == 200
        ids.add(resp.json()["job_id"])

    listing = client.get("/jobs")
    assert listing.status_code == 200
    body = listing.json()
    listed_ids = {j["job_id"] for j in body}
    assert ids <= listed_ids
    # The flattened params (lat/lon for trap positions, status for queue) survive.
    sample = next(j for j in body if j["job_id"] in ids)
    assert sample["status"] == "done"
    assert "lat" in sample and "created_at" in sample


# ---------------------------------------------------------------------------
# Helpers for detection/clip/label routes (no patched pipeline — real files)
# ---------------------------------------------------------------------------


def _seed_job_with_detections(
    jobs_root: Path, n: int = 2
) -> tuple[str, list[Detection]]:
    """Create a real job dir with detections.jsonl + segments/<id>.wav clips."""
    job = jobs.create_job(jobs_root, params={"source_path": "/data/A1"})
    job_dir = job.workdir
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)

    dets: list[Detection] = []
    for i in range(n):
        seg = Segment(start_s=1.0 + i, duration_s=0.5)
        det = Detection.new(
            trap_id="A1",
            source_file="REC.wav",
            segment=seg,
            labels=[
                Label.from_prediction(
                    type("P", (), {"species": "Turdus merula", "probability": 0.9})(),
                    source=SOURCE_STUB,
                    status=STATUS_PSEUDO,
                )
            ],
        )
        clip = 0.01 * np.random.default_rng(i).standard_normal((48000 // 2, 2))
        sf.write(job_dir / det.segment_path, clip, 48000)
        dets.append(det)

    write_detections(job_dir / "detections.jsonl", dets)
    return str(job.job_id), dets


def test_get_detections_returns_seeded(client, tmp_path):
    job_id, dets = _seed_job_with_detections(tmp_path / "jobs")
    resp = client.get(f"/jobs/{job_id}/detections")
    assert resp.status_code == 200
    body = resp.json()
    assert {d["detection_id"] for d in body} == {d.detection_id for d in dets}
    assert all(d["localization"] is None for d in body)


def test_get_detections_missing_file_is_empty_list(client, tmp_path):
    """A job that ran before this feature (no detections.jsonl) -> [] not 500."""
    job = jobs.create_job(tmp_path / "jobs", params={"source_path": "/data/A1"})
    resp = client.get(f"/jobs/{job.job_id}/detections")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_detections_unknown_job_404(client):
    resp = client.get("/jobs/does-not-exist/detections")
    assert resp.status_code == 404


def test_get_segment_clip_ok(client, tmp_path):
    job_id, dets = _seed_job_with_detections(tmp_path / "jobs")
    det_id = dets[0].detection_id
    resp = client.get(f"/jobs/{job_id}/segments/{det_id}.wav")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_get_segment_rejects_traversal(client, tmp_path):
    job_id, _dets = _seed_job_with_detections(tmp_path / "jobs")
    # Non-hex id with traversal chars must never escape the segments dir.
    resp = client.get(f"/jobs/{job_id}/segments/..%2F..%2Fmanifest.wav")
    assert resp.status_code in (400, 404)
    resp2 = client.get(f"/jobs/{job_id}/segments/not-a-real-id.wav")
    assert resp2.status_code == 404


def test_post_label_appends_corrected(client, tmp_path):
    job_id, dets = _seed_job_with_detections(tmp_path / "jobs")
    det_id = dets[0].detection_id
    resp = client.post(
        f"/jobs/{job_id}/detections/{det_id}/label",
        json={"species": "Erithacus rubecula"},
    )
    assert resp.status_code == 200

    reread = read_detections(tmp_path / "jobs" / job_id / "detections.jsonl")
    target = next(d for d in reread if d.detection_id == det_id)
    new_label = target.labels[-1]
    assert new_label.species == "Erithacus rubecula"
    assert new_label.source == "operator:ranger"
    assert new_label.status == "corrected"
    assert new_label.probability is None


def test_post_label_unknown_detection_404(client, tmp_path):
    job_id, _dets = _seed_job_with_detections(tmp_path / "jobs")
    resp = client.post(
        f"/jobs/{job_id}/detections/deadbeefdeadbeefdeadbeefdeadbeef/label",
        json={"species": "X"},
    )
    assert resp.status_code == 404


def test_concurrent_label_no_lost_update(client, tmp_path):
    """Two concurrent POSTs adding DIFFERENT species must both survive (flock RMW)."""
    job_id, dets = _seed_job_with_detections(tmp_path / "jobs")
    det_id = dets[0].detection_id
    base_labels = len(dets[0].labels)

    def post(species: str):
        return client.post(
            f"/jobs/{job_id}/detections/{det_id}/label",
            json={"species": species},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(post, "Species one")
        f2 = pool.submit(post, "Species two")
        r1, r2 = f1.result(), f2.result()
    assert r1.status_code == 200 and r2.status_code == 200

    reread = read_detections(tmp_path / "jobs" / job_id / "detections.jsonl")
    target = next(d for d in reread if d.detection_id == det_id)
    species_added = {lab.species for lab in target.labels[base_labels:]}
    assert species_added == {"Species one", "Species two"}, (
        "lost-update: both concurrent labels must be present"
    )
    assert len(target.labels) == base_labels + 2
