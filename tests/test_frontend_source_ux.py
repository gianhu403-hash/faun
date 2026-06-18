"""Frontend source-type UX backstop (Vector E).

The single source input (folder | URL | Yandex.Disk) must give the operator
honest feedback: what *kind* of source it is, a download-vs-process distinction
while a remote job runs, and a human-readable RU label for the backend's new
``error_kind`` field. These are pure additive affordances on top of the frozen
3-window UI, so this test also re-asserts the invariants the existing render
tests rely on (so a regression here is caught locally, not only in CI).

TF-free and network-free: drives ``faun.api.app`` through ``TestClient`` and
inspects the *served* HTML/JS text. No browser, no bundler — the UI is vanilla
JS, so the source-type logic and the error_kind mapping live as literal strings
in ``app.js`` / ``index.html`` and are assertable by substring.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from faun import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with FAUN_JOBS_ROOT pointed at a temp dir (no real jobs)."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    return TestClient(api.app)


@pytest.fixture
def app_js(client) -> str:
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def index_html(client) -> str:
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


# ---------------------------------------------------------------------------
# Preserved invariants — keep the existing render tests green from here too.
# ---------------------------------------------------------------------------


def test_preserved_index_invariants(index_html: str) -> None:
    assert "Faun" in index_html
    assert "/static/app.js" in index_html
    assert "/static/styles.css" in index_html
    # The POST body shape and core form controls are untouched.
    assert 'id="job-form"' in index_html
    assert 'id="source"' in index_html
    assert 'id="submit-btn"' in index_html


def test_post_body_shape_unchanged(client, monkeypatch) -> None:
    """The contract stays {source_path, lat, lon}: the additive UX must not
    change what the form actually submits."""
    captured: list = []

    def fake_run_pipeline(job_dir, source_path, lat=None, lon=None, classifier=None):
        from pathlib import Path

        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        results = job_dir / "results.csv"
        results.write_text(
            "track,start_sec,duration_sec,species,probability\n", "utf-8"
        )
        captured.append((source_path, lat, lon))
        return results

    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)
    resp = client.post(
        "/jobs", json={"source_path": "/data/A1", "lat": 1.0, "lon": 2.0}
    )
    assert resp.status_code == 200
    assert captured == [("/data/A1", 1.0, 2.0)]


# ---------------------------------------------------------------------------
# Source-type hint affordance: sourceKind() + the three RU labels.
# ---------------------------------------------------------------------------


def test_source_kind_helper_present(app_js: str) -> None:
    """The served app.js exposes a sourceKind(value) classifier helper."""
    assert "function sourceKind" in app_js


def test_source_kind_three_labels_present(app_js: str) -> None:
    """All three human RU hints ship in the served JS."""
    assert "папка" in app_js
    assert "URL" in app_js
    assert "Яндекс.Диск" in app_js


def test_source_kind_classification_keys_present(app_js: str) -> None:
    """The three logical kinds folder|url|yadisk are the helper's vocabulary,
    and the Yandex.Disk hosts it keys off are present in the served JS."""
    for kind in ("folder", "url", "yadisk"):
        assert kind in app_js, f"missing source-kind key: {kind}"
    # yadisk detection keys off the canonical hosts.
    assert "disk.yandex.ru" in app_js
    assert "yadi.sk" in app_js


def test_index_has_hint_affordance(index_html: str, app_js: str) -> None:
    """A hint element near the source input exists (id referenced by app.js)."""
    assert 'id="source-hint"' in index_html
    assert "source-hint" in app_js


# ---------------------------------------------------------------------------
# Download-vs-process honesty for remote sources.
# ---------------------------------------------------------------------------


def test_download_vs_process_distinction(app_js: str) -> None:
    """A running remote job distinguishes 'downloading source' from 'processing'."""
    assert "Скачивание источника" in app_js
    assert "Обработка" in app_js


# ---------------------------------------------------------------------------
# error_kind -> RU label mapping.
# ---------------------------------------------------------------------------


def test_error_kind_mapping_present(app_js: str) -> None:
    """The served app.js maps backend error_kind values to RU labels."""
    assert "errorKindLabel" in app_js or "ERROR_KIND_LABELS" in app_js


@pytest.mark.parametrize(
    "kind",
    [
        "ssrf",
        "too-large",
        "not-found",
        "zip-slip",
        "not-an-archive",
        "bad-scheme",
        "network",
        "empty",
    ],
)
def test_error_kind_keys_present(app_js: str, kind: str) -> None:
    """Each error_kind the backend may emit has a mapping entry."""
    assert kind in app_js, f"missing error_kind entry: {kind}"


def test_error_kind_ru_labels_present(app_js: str) -> None:
    """A couple of the human RU labels ship verbatim (honest, not codes)."""
    assert "источник заблокирован" in app_js
    assert "архив слишком большой" in app_js
    assert "источник не найден" in app_js
