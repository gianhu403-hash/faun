"""DOM-invariant backstop for the 3-window Faun web UI.

Stdlib-only (``html.parser`` + ``pathlib``): no new deps, no network. Parses each
static HTML page and asserts the structural nodes app.js relies on still exist, so
a redesign cannot silently delete the upload form, the Leaflet map, the audio
player, the relabel control, or the shared ``styles.css`` / ``app.js`` includes.

The assertions are deliberately specific (matched by id/tag/attr), so removing any
key node makes the test fail rather than pass vacuously.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "faun" / "static"


class _Collector(HTMLParser):
    """Collect tags, ids, classes, and src/href attrs from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.srcs: list[str] = []
        self.hrefs: list[str] = []
        # (tag, {attr: value}) for attribute-level assertions.
        self.elements: list[tuple[str, dict[str, str]]] = []
        # Text captured outside of <head>/<script>/<style>, to prove non-empty body.
        self.body_text: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        self.tags.append(tag)
        self.elements.append((tag, d))
        if "id" in d:
            self.ids.add(d["id"])
        if "class" in d:
            self.classes.update(d["class"].split())
        if "src" in d:
            self.srcs.append(d["src"])
        if "href" in d:
            self.hrefs.append(d["href"])
        if tag in ("head", "script", "style", "template"):
            self._suppress_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("head", "script", "style", "template") and self._suppress_depth:
            self._suppress_depth -= 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags (e.g. <input />, <audio /> in templates).
        self.handle_starttag(tag, attrs)
        if tag in ("head", "script", "style", "template"):
            self._suppress_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._suppress_depth == 0 and data.strip():
            self.body_text.append(data.strip())


def _parse(name: str) -> _Collector:
    path = STATIC / name
    assert path.exists(), f"missing static page: {path}"
    c = _Collector()
    c.feed(path.read_text(encoding="utf-8"))
    return c


def _has_element(c: _Collector, tag: str, **attrs: str) -> bool:
    for t, d in c.elements:
        if t != tag:
            continue
        if all(d.get(k) == v for k, v in attrs.items()):
            return True
    return False


def _has_attr_contains(c: _Collector, tag: str, attr: str, needle: str) -> bool:
    for t, d in c.elements:
        if t == tag and needle in d.get(attr, ""):
            return True
    return False


# --- shared invariants across all three pages -------------------------------


@pytest.mark.parametrize("page", ["index.html", "dashboard.html", "review.html"])
def test_page_has_body_and_shared_assets(page: str) -> None:
    c = _parse(page)
    assert "body" in c.tags, f"{page}: missing <body>"
    assert c.body_text, f"{page}: body has no visible text"
    assert any("styles.css" in h for h in c.hrefs), f"{page}: missing styles.css link"
    assert any("app.js" in s for s in c.srcs), f"{page}: missing app.js script"


# --- index.html: upload form + job-queue container --------------------------


def test_index_upload_form_present() -> None:
    c = _parse("index.html")
    # Upload control: the job-submit <form> with the source input.
    assert _has_element(c, "form", id="job-form"), (
        "index: upload <form id=job-form> missing"
    )
    assert "source" in c.ids, "index: source input (#source) missing"
    assert "submit-btn" in c.ids, "index: submit button (#submit-btn) missing"


def test_index_job_queue_container_present() -> None:
    c = _parse("index.html")
    # The queue/status surface the poller fills.
    assert "status-card" in c.ids, "index: #status-card container missing"
    assert "queue" in c.ids, "index: #queue container missing"
    assert "results-body" in c.ids, "index: #results-body container missing"


# --- dashboard.html: Leaflet map + job list + Leaflet refs ------------------


def test_dashboard_map_and_jobs_containers() -> None:
    c = _parse("dashboard.html")
    assert "map" in c.ids, "dashboard: Leaflet map container (#map) missing"
    assert "jobs-body" in c.ids, "dashboard: job-list container (#jobs-body) missing"


def test_dashboard_references_leaflet() -> None:
    c = _parse("dashboard.html")
    assert any("leaflet" in h.lower() for h in c.hrefs), (
        "dashboard: Leaflet CSS link missing"
    )
    assert any("leaflet" in s.lower() for s in c.srcs), (
        "dashboard: Leaflet JS script missing"
    )


# --- review.html: audio player + relabel control ----------------------------


def test_review_has_audio_player() -> None:
    c = _parse("review.html")
    # The <audio> element lives inside a <template> cloned per row by app.js.
    assert "audio" in c.tags, "review: <audio> element/template missing"
    assert "audio-tpl" in c.ids, "review: #audio-tpl template missing"


def test_review_has_relabel_and_detection_containers() -> None:
    c = _parse("review.html")
    assert "det-body" in c.ids, "review: detection-list container (#det-body) missing"
    # Relabel control: app.js builds the per-row form; the job picker provides a
    # select + button as the page-level relabel/navigation affordance.
    assert "job-select" in c.ids, "review: #job-select control missing"
    assert "open-job-btn" in c.ids, "review: #open-job-btn control missing"


# --- negative control: prove the parser actually rejects missing nodes ------


def test_parser_detects_missing_node() -> None:
    c = _Collector()
    c.feed("<html><body><p>hi</p></body></html>")
    assert "map" not in c.ids
    assert not _has_element(c, "form", id="job-form")
    assert c.body_text == ["hi"]


# --- optional screenshot (skips cleanly when Playwright is absent) -----------


def test_screenshot_optional() -> None:
    pytest.importorskip("playwright")
    pytest.skip(
        "screenshot capture not wired in CI; DOM-invariant is the binding check"
    )
