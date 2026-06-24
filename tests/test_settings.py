"""Tests for faun.settings — the single typed source of truth for FAUN_* config.

Pure stdlib: no heavy imports. These tests REALLY exercise the parser and the
cache (no mock-only paths): they monkeypatch ``os.environ`` and assert that
``Settings.from_env`` / ``get_settings`` reflect the live environment, that
typed defaults kick in when a var is unset, that malformed integers fall back
to the documented default rather than crashing, and that the cached accessor
is invalidatable so per-test env overrides are honoured.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import faun.settings as settings_mod
from faun.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Drop the cached Settings before and after every test.

    ``get_settings`` is cached on purpose, so without this other tests' env
    mutations would leak through the singleton. Clearing on both sides keeps
    each test hermetic.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_when_env_empty(monkeypatch) -> None:
    """With no FAUN_* / model-path vars set, every field is its documented default."""
    for var in (
        "FAUN_JOBS_ROOT",
        "FAUN_CLASSIFIER",
        "FAUN_SOURCE_TIMEOUT_S",
        "FAUN_SOURCE_MAX_BYTES",
        "FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES",
        "FAUN_SOURCE_MAX_ENTRIES",
        "FAUN_SOURCE_MAX_REDIRECTS",
        "FAUN_LOG_JSON",
        "PERCH_V2_MODEL_PATH",
        "PERCH_MODEL_PATH",
        "YAMNET_PROBE_PATH",
        "FAUN_SPECIES_ALLOWLIST",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings.from_env()

    assert s.jobs_root == Path("./jobs")
    assert s.classifier == "stub"
    # Source-limit defaults mirror faun.sources' authoritative (enforced) values,
    # sized for real ~23 GB Yandex.Disk trap folders.
    assert s.timeout_s == 60.0
    assert s.max_bytes == 30 * 1024**3
    assert s.max_uncompressed_bytes == 60 * 1024**3
    assert s.max_entries == 100_000
    assert s.max_redirects == 5
    assert s.log_json is True
    assert s.perch_v2_model_path is None
    assert s.perch_model_path is None
    assert s.yamnet_probe_path is None
    assert s.species_allowlist is None


# ---------------------------------------------------------------------------
# Typed parsing of each var
# ---------------------------------------------------------------------------


def test_reads_each_var(monkeypatch) -> None:
    """Every FAUN_* / model knob is read from the environment with its type."""
    monkeypatch.setenv("FAUN_JOBS_ROOT", "/srv/faun/jobs")
    monkeypatch.setenv("FAUN_CLASSIFIER", "  Perch  ")  # trimmed + lowered
    monkeypatch.setenv("FAUN_SOURCE_TIMEOUT_S", "12.5")
    monkeypatch.setenv("FAUN_SOURCE_MAX_BYTES", "1048576")
    monkeypatch.setenv("FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", "2097152")
    monkeypatch.setenv("FAUN_SOURCE_MAX_ENTRIES", "42")
    monkeypatch.setenv("FAUN_SOURCE_MAX_REDIRECTS", "9")
    monkeypatch.setenv("FAUN_LOG_JSON", "0")
    monkeypatch.setenv("PERCH_V2_MODEL_PATH", "/models/perch2")
    monkeypatch.setenv("PERCH_MODEL_PATH", "/models/perch1")
    monkeypatch.setenv("YAMNET_PROBE_PATH", "/models/probe.pkl")
    monkeypatch.setenv("FAUN_SPECIES_ALLOWLIST", "/etc/faun/reserve.txt")

    s = Settings.from_env()

    assert s.jobs_root == Path("/srv/faun/jobs")
    assert s.classifier == "perch"
    assert s.timeout_s == 12.5
    assert s.max_bytes == 1_048_576
    assert s.max_uncompressed_bytes == 2_097_152
    assert s.max_entries == 42
    assert s.max_redirects == 9
    assert s.log_json is False
    assert s.perch_v2_model_path == "/models/perch2"
    assert s.perch_model_path == "/models/perch1"
    assert s.yamnet_probe_path == "/models/probe.pkl"
    assert s.species_allowlist == "/etc/faun/reserve.txt"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", True),  # empty -> default (json on)
        ("garbage", True),  # unrecognised -> default (json on)
    ],
)
def test_log_json_truthy_parsing(monkeypatch, value, expected) -> None:
    """FAUN_LOG_JSON accepts common bool spellings; unknowns fall back to default."""
    monkeypatch.setenv("FAUN_LOG_JSON", value)
    assert Settings.from_env().log_json is expected


# ---------------------------------------------------------------------------
# Defensive int/float parsing
# ---------------------------------------------------------------------------


def test_malformed_int_falls_back_to_default(monkeypatch) -> None:
    """A non-numeric int var must NOT crash boot — it falls back to the default."""
    monkeypatch.setenv("FAUN_SOURCE_MAX_ENTRIES", "not-a-number")
    monkeypatch.setenv("FAUN_SOURCE_MAX_REDIRECTS", "")
    monkeypatch.setenv("FAUN_SOURCE_MAX_BYTES", "12.5")  # float string is not an int

    s = Settings.from_env()

    assert s.max_entries == 100_000
    assert s.max_redirects == 5
    assert s.max_bytes == 30 * 1024**3


def test_malformed_float_falls_back_to_default(monkeypatch) -> None:
    """A non-numeric float var falls back to the documented default."""
    monkeypatch.setenv("FAUN_SOURCE_TIMEOUT_S", "soon")
    assert Settings.from_env().timeout_s == 60.0


def test_negative_and_zero_limits_fall_back(monkeypatch) -> None:
    """Non-positive limits are nonsensical and must fall back to the default."""
    monkeypatch.setenv("FAUN_SOURCE_MAX_ENTRIES", "0")
    monkeypatch.setenv("FAUN_SOURCE_MAX_BYTES", "-5")
    monkeypatch.setenv("FAUN_SOURCE_TIMEOUT_S", "-1")

    s = Settings.from_env()

    assert s.max_entries == 100_000
    assert s.max_bytes == 30 * 1024**3
    assert s.timeout_s == 60.0


def test_blank_string_paths_become_none(monkeypatch) -> None:
    """An empty/whitespace model-path var is treated as unset (None)."""
    monkeypatch.setenv("PERCH_V2_MODEL_PATH", "   ")
    monkeypatch.setenv("PERCH_MODEL_PATH", "")
    s = Settings.from_env()
    assert s.perch_v2_model_path is None
    assert s.perch_model_path is None


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


def test_settings_is_frozen() -> None:
    """Settings is immutable — attribute assignment raises FrozenInstanceError."""
    s = Settings.from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.classifier = "perch"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cached accessor + invalidation
# ---------------------------------------------------------------------------


def test_get_settings_is_cached(monkeypatch) -> None:
    """get_settings returns the SAME instance on repeated calls (cached)."""
    monkeypatch.delenv("FAUN_CLASSIFIER", raising=False)
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_cache_can_be_cleared(monkeypatch) -> None:
    """Clearing the cache lets a freshly-monkeypatched env be picked up.

    This is the behaviour other tests rely on: they set FAUN_* per-test, so the
    singleton MUST be invalidatable, otherwise the first test to call
    get_settings would freeze the config for the whole session.
    """
    monkeypatch.setenv("FAUN_CLASSIFIER", "stub")
    assert get_settings().classifier == "stub"

    monkeypatch.setenv("FAUN_CLASSIFIER", "perch")
    # Stale cache still says stub until invalidated.
    assert get_settings().classifier == "stub"

    get_settings.cache_clear()
    assert get_settings().classifier == "perch"
