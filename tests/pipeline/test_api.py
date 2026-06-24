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


def test_get_segment_spectrogram_ok(client, tmp_path):
    """FR-008/SC-E1: the .png route renders a spectrogram and caches it."""
    job_id, dets = _seed_job_with_detections(tmp_path / "jobs")
    det_id = dets[0].detection_id
    resp = client.get(f"/jobs/{job_id}/segments/{det_id}.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes
    # Second request hits the cached file (still a valid PNG).
    again = client.get(f"/jobs/{job_id}/segments/{det_id}.png")
    assert again.status_code == 200
    assert again.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_segment_spectrogram_rejects_traversal(client, tmp_path):
    job_id, _dets = _seed_job_with_detections(tmp_path / "jobs")
    # Same hex-id guard as the .wav route (SC-E1).
    resp = client.get(f"/jobs/{job_id}/segments/..%2F..%2Fsecret.png")
    assert resp.status_code in (400, 404)
    resp2 = client.get(f"/jobs/{job_id}/segments/not-a-real-id.png")
    assert resp2.status_code == 404


def test_get_segment_spectrogram_missing_clip_404(client, tmp_path):
    job_id, _dets = _seed_job_with_detections(tmp_path / "jobs")
    # A well-formed but unknown hex id has no clip -> 404, not a 500.
    resp = client.get(f"/jobs/{job_id}/segments/{'a' * 32}.png")
    assert resp.status_code == 404


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


# ---------------------------------------------------------------------------
# Wave-2 P0 e2e: a URL / Yandex.Disk source resolved through the REAL pipeline
# (run_pipeline is NOT patched here; httpx + DNS are mocked, no network).
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _zip_tree_bytes(root: Path) -> bytes:
    """Zip a directory tree to in-memory bytes (arcnames relative to root)."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(root).as_posix())
    return buf.getvalue()


class _FakeStream:
    """httpx streaming-response stand-in for the download leg."""

    def __init__(self, url: str, payload: bytes) -> None:
        self.url = url
        self.status_code = 200
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size: int = 65536):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


class _FakeResp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeYandexClient:
    """Minimal httpx.Client stand-in: resolve href -> stream a ZIP. No network."""

    def __init__(self, zip_bytes: bytes, **_kw) -> None:
        self._zip = zip_bytes

    def get(self, url, params=None):
        # /resources/download -> a download href on an allowlisted Yandex host.
        return _FakeResp({"href": "https://downloader.disk.yandex.ru/disk/zip?x=1"})

    def stream(self, method, href):
        return _FakeStream("https://s1.storage.yandex.net/rdisk/A1.zip", self._zip)

    def close(self):
        return None


def _patch_public_dns(monkeypatch):
    """Make every host resolve to a public IP so the SSRF guard passes offline."""
    import faun.sources as sources

    def fake_getaddrinfo(host, *a, **k):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]  # AF_INET, public IP

    monkeypatch.setattr(sources.socket, "getaddrinfo", fake_getaddrinfo)


def test_p0_yandex_url_resolved_through_pipeline(client, tmp_path, monkeypatch):
    """P0: a Yandex.Disk URL is downloaded+extracted and ingested for real.

    Before the fix run_pipeline did Path("https://...") -> ingest FileNotFound.
    Now resolve_source fetches a ZIP (mocked httpx) -> A1/A2 -> results.csv, and
    ingest.scan never sees an "https:/"-mangled path.
    """
    import json

    import faun.sources as sources

    zip_bytes = _zip_tree_bytes(_FIXTURES / "traps_mini")
    _patch_public_dns(monkeypatch)
    monkeypatch.setattr(
        sources.httpx, "Client", lambda **kw: _FakeYandexClient(zip_bytes, **kw)
    )

    resp = client.post("/jobs", json={"url": "https://disk.yandex.ru/d/TESTKEY"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "done", body  # resolve+ingest+classify+write all ran
    csv = client.get(f"/jobs/{job_id}/results.csv")
    assert csv.status_code == 200

    meta = json.loads(
        (tmp_path / "jobs" / job_id / "results_meta.json").read_text(encoding="utf-8")
    )
    assert meta["source_provenance"]["mode"] == "yadisk"
    assert meta["source_provenance"]["source"] == "https://disk.yandex.ru/d/TESTKEY"
    # ingest saw BOTH traps (A1 + A2) from the extracted archive.
    assert len(meta["files"]) == 2


def test_p0_resolve_failure_marks_job_error_not_500(client):
    """An SSRF/internal source -> job status='error' with error_kind, never 500."""
    resp = client.post("/jobs", json={"url": "http://127.0.0.1/evil.zip"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "error"
    assert body.get("error_kind") == "ssrf"
    assert "error" in body


# ---------------------------------------------------------------------------
# B1 disk-leak: the downloaded/extracted REMOTE source tree is cleaned up
# ---------------------------------------------------------------------------


def test_remote_source_dir_cleaned_after_job(client, tmp_path, monkeypatch):
    """A remote (downloaded+extracted) source tree is removed after the job.

    job_dir/_source must NOT survive, while the real outputs (results.csv + the
    segment clips) remain. Same mocked-httpx Yandex path as the P0 e2e test.
    """
    import faun.sources as sources

    zip_bytes = _zip_tree_bytes(_FIXTURES / "traps_mini")
    _patch_public_dns(monkeypatch)
    monkeypatch.setattr(
        sources.httpx, "Client", lambda **kw: _FakeYandexClient(zip_bytes, **kw)
    )

    resp = client.post("/jobs", json={"url": "https://disk.yandex.ru/d/CLEANUP"})
    job_id = resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "done"

    job_dir = tmp_path / "jobs" / job_id
    assert not (job_dir / "_source").exists(), "remote _source tree must be cleaned up"
    assert (job_dir / "results.csv").exists()
    assert list((job_dir / "segments").glob("*.wav")), "clips must survive cleanup"


def test_local_source_not_deleted_by_pipeline(client, tmp_path):
    """A LOCAL pass-through source (outside job_dir) is never touched by cleanup."""
    resp = client.post("/jobs", json={"source_path": str(_FIXTURES / "traps_mini")})
    job_id = resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "done"
    # The committed fixture dir must remain intact (cleanup only removes _source).
    assert (_FIXTURES / "traps_mini").is_dir()
    assert not (tmp_path / "jobs" / job_id / "_source").exists()


# ---------------------------------------------------------------------------
# B3 input validation: lat/lon bounds + label length (422, never poisoned data)
# ---------------------------------------------------------------------------


def test_job_rejects_out_of_range_lat(client):
    resp = client.post("/jobs", json={"source_path": "/data/A1", "lat": 200.0})
    assert resp.status_code == 422


def test_jobrequest_rejects_non_finite_coords():
    """JobRequest rejects NaN / +-Inf coords at the model level (the real gate).

    Tested on the model rather than over HTTP: a standards-compliant JSON client
    cannot even encode NaN, and FastAPI's own 422 body can't serialise it either.
    The bound check stops a non-finite coordinate from poisoning results_meta.json.
    """
    from pydantic import ValidationError

    from faun.api import JobRequest

    for bad in (float("nan"), float("inf"), float("-inf"), 200.0, -100.0):
        with pytest.raises(ValidationError):
            JobRequest(source_path="/data/A1", lat=bad)
    # A valid in-range coordinate still constructs fine.
    assert JobRequest(source_path="/data/A1", lat=58.1, lon=45.2).lat == 58.1


# ---------------------------------------------------------------------------
# Classifier build + regional allow-list wiring (FR-001, ADR-0004)
# ---------------------------------------------------------------------------


class TestBuildClassifierAllowlist:
    def test_unmasked_by_default(self, monkeypatch) -> None:
        """No FAUN_SPECIES_ALLOWLIST -> the bare classifier (prod-safe path)."""
        from faun.classification import StubAdapter
        from faun.settings import get_settings

        monkeypatch.delenv("FAUN_SPECIES_ALLOWLIST", raising=False)
        monkeypatch.setenv("FAUN_CLASSIFIER", "stub")
        get_settings.cache_clear()
        clf = api._build_classifier()
        assert isinstance(clf, StubAdapter)

    def test_wraps_when_allowlist_set(self, monkeypatch) -> None:
        from faun.classification import MaskedClassifier, StubAdapter
        from faun.settings import get_settings

        monkeypatch.setenv("FAUN_CLASSIFIER", "stub")
        monkeypatch.setenv("FAUN_SPECIES_ALLOWLIST", "default")
        get_settings.cache_clear()
        clf = api._build_classifier()
        assert isinstance(clf, MaskedClassifier)
        assert isinstance(clf.inner, StubAdapter)

    def test_missing_file_stays_unmasked(self, monkeypatch, tmp_path) -> None:
        from faun.classification import StubAdapter
        from faun.settings import get_settings

        monkeypatch.setenv("FAUN_CLASSIFIER", "stub")
        monkeypatch.setenv("FAUN_SPECIES_ALLOWLIST", str(tmp_path / "nope.txt"))
        get_settings.cache_clear()
        clf = api._build_classifier()
        assert isinstance(clf, StubAdapter)  # bad file -> no wrap (no-op)

    def test_classifier_source_unwraps_mask(self) -> None:
        from faun.classification import MaskedClassifier, StubAdapter
        from faun.detections import SOURCE_STUB

        masked = MaskedClassifier(StubAdapter(), ["Turdus merula"])
        assert api._classifier_source(masked) == SOURCE_STUB
