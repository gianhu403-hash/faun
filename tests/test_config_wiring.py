"""Tests that config readers go through faun.settings.get_settings (ADR-0001).

These assert the *wiring*, not the underlying parser (that lives in
``tests/test_settings.py``): ``faun.sources`` reads its hardening limits from
``get_settings()``, the Perch / YAMNet adapters resolve their model paths from
``get_settings()`` (with the constructor argument keeping priority), and
``faun.health`` reads ``jobs_root`` from ``get_settings()``.

Pure stdlib + numpy/soxr/httpx — no TensorFlow / PyTorch import. The autouse
fixture in ``tests/conftest.py`` clears the ``get_settings`` lru_cache around
each test; where a test sets the env *mid-test* it clears the cache itself so
the freshly-set value is read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faun import sources
from faun.classification.perch import DEFAULT_TFHUB_URL, PerchAdapter
from faun.classification.yamnet import YAMNetAdapter
from faun.health import health
from faun.settings import get_settings


# ---------------------------------------------------------------------------
# faun.sources reads limits from get_settings()
# ---------------------------------------------------------------------------


def test_sources_reads_max_redirects_from_settings(monkeypatch) -> None:
    """A FAUN_SOURCE_* override flows into faun.sources via get_settings()."""
    monkeypatch.setenv("FAUN_SOURCE_MAX_REDIRECTS", "1")
    get_settings.cache_clear()
    # The value faun.sources will read is sourced from Settings.
    assert get_settings().max_redirects == 1


def test_sources_enforces_settings_redirect_cap(monkeypatch) -> None:
    """resolve_source honours the get_settings()-sourced redirect cap.

    With a cap of 0, a single redirect hop must exhaust the budget and fail
    with ``kind="network"`` (too-many-redirects), proving _stream_to_file reads
    the limit from Settings rather than a module constant.
    """
    import socket

    monkeypatch.setattr(
        sources.socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )
    monkeypatch.setenv("FAUN_SOURCE_MAX_REDIRECTS", "0")
    get_settings.cache_clear()

    class _AlwaysRedirect:
        def __init__(self) -> None:
            self.stream_calls: list[str] = []

        def stream(self, method: str, url: str, **kw):
            self.stream_calls.append(url)

            class _Resp:
                status_code = 302
                headers = {"Location": "https://cdn.example.org/next.zip"}
                url = url

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return None

            return _Resp()

    client = _AlwaysRedirect()
    with pytest.raises(sources.SourceError) as exc:
        sources.resolve_source(
            "https://files.example.com/traps.zip", Path("/tmp/_wd_redir"), client=client
        )
    assert exc.value.kind == "network"
    # max_redirects=0 => only the initial hop is issued before the budget is spent.
    assert client.stream_calls == ["https://files.example.com/traps.zip"]


# ---------------------------------------------------------------------------
# PerchAdapter model-path resolution via settings
# ---------------------------------------------------------------------------


def test_perch_picks_up_env_model_path_via_settings(monkeypatch) -> None:
    monkeypatch.setenv("PERCH_MODEL_PATH", "/models/perch-from-env")
    get_settings.cache_clear()
    adapter = PerchAdapter(model_path=None)
    assert adapter.model_path == "/models/perch-from-env"


def test_perch_constructor_arg_overrides_settings(monkeypatch) -> None:
    monkeypatch.setenv("PERCH_MODEL_PATH", "/models/perch-from-env")
    get_settings.cache_clear()
    adapter = PerchAdapter(model_path="/explicit/perch")
    assert adapter.model_path == "/explicit/perch"


def test_perch_default_tfhub_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("PERCH_MODEL_PATH", raising=False)
    get_settings.cache_clear()
    adapter = PerchAdapter(model_path=None)
    assert adapter.model_path == DEFAULT_TFHUB_URL


# ---------------------------------------------------------------------------
# YAMNetAdapter probe-path resolution via settings
# ---------------------------------------------------------------------------


def test_yamnet_picks_up_env_probe_path_via_settings(monkeypatch) -> None:
    monkeypatch.setenv("YAMNET_PROBE_PATH", "/models/probe-from-env.pkl")
    get_settings.cache_clear()
    adapter = YAMNetAdapter()
    assert adapter.probe_path == "/models/probe-from-env.pkl"


def test_yamnet_constructor_arg_overrides_settings(monkeypatch) -> None:
    monkeypatch.setenv("YAMNET_PROBE_PATH", "/models/probe-from-env.pkl")
    get_settings.cache_clear()
    adapter = YAMNetAdapter(probe_path="/explicit/probe.pkl")
    assert adapter.probe_path == "/explicit/probe.pkl"


def test_yamnet_probe_path_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("YAMNET_PROBE_PATH", raising=False)
    get_settings.cache_clear()
    adapter = YAMNetAdapter()
    assert adapter.probe_path is None


# ---------------------------------------------------------------------------
# faun.health reads jobs_root from settings
# ---------------------------------------------------------------------------


def test_health_jobs_root_from_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_JOBS_ROOT", str(tmp_path / "jobs"))
    get_settings.cache_clear()
    payload = health()
    assert payload["jobs_root_writable"] is True
    assert payload["status"] == "ok"
    # the dir resolved from Settings was actually created
    assert (tmp_path / "jobs").is_dir()
