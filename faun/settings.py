"""Centralized, typed configuration for Faun — one home for all FAUN_* knobs.

Before this module, ``FAUN_*`` (and the model-path) environment variables were
read ad-hoc and inconsistently: ``faun.api`` resolved ``FAUN_JOBS_ROOT`` /
``FAUN_CLASSIFIER`` inline, ``faun.health`` re-implemented the same jobs-root
resolution, and each adapter (``PERCH_MODEL_PATH``, ``YAMNET_PROBE_PATH``) read
its own ``os.environ`` call. The forthcoming source-resolution layer
(``faun.sources`` — see ``docs/adr/0001-source-resolution-layer.md``) adds a set
of SSRF / zip-bomb hardening limits that also need a single, defensible home.

``Settings`` is a frozen dataclass parsed once from the environment via
``from_env``. ``get_settings`` returns a cached singleton; it is intentionally
invalidatable (``get_settings.cache_clear()``) so tests can monkeypatch the
environment per-test and pick up the change.

Defensive parsing is a deliberate enterprise choice: a malformed or
non-positive numeric env var must NOT crash service boot. Such values fall back
to the documented default and the misconfiguration is surfaced via a log
warning rather than a traceback.

WIRING (current): ``get_settings()`` is now the actual reader across the
service. ``faun.api`` consumes ``jobs_root`` / ``classifier`` / ``log_json`` /
``basic_user`` / ``basic_pass``; ``faun.sources`` reads ``timeout_s`` /
``max_bytes`` / ``max_redirects`` / ``max_entries`` / ``max_uncompressed_bytes``;
``faun.health`` reads ``jobs_root``; the model adapters resolve their paths via
``perch_model_path`` / ``perch_v2_model_path`` / ``yamnet_probe_path`` (an
explicit constructor argument still wins). One deliberate exception remains:
``faun.sources`` reads the uncompressed-cap through its ``_int_env`` indirection
(whose default is sourced from ``max_uncompressed_bytes``) so a direct
``FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES`` env override still wins at call time and
the zip-bomb enforcement test keeps its patch point.

Defaults (match the values faun.sources actually enforces):
    jobs_root                 ``./jobs``      (FAUN_JOBS_ROOT)         [WIRED]
    classifier                ``stub``        (FAUN_CLASSIFIER)        [WIRED]
    timeout_s                 ``60.0`` s      (FAUN_SOURCE_TIMEOUT_S)
    max_bytes                 ``30 GiB``      (FAUN_SOURCE_MAX_BYTES) — download cap
    max_uncompressed_bytes    ``60 GiB``      (FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES) — zip-bomb cap
    max_entries               ``100000``      (FAUN_SOURCE_MAX_ENTRIES) — archive member cap
    max_redirects             ``5``           (FAUN_SOURCE_MAX_REDIRECTS) — SSRF redirect cap
    log_json                  ``True``        (FAUN_LOG_JSON)          [WIRED]
    perch_v2_model_path       ``None``        (PERCH_V2_MODEL_PATH)
    perch_model_path          ``None``        (PERCH_MODEL_PATH)
    yamnet_probe_path         ``None``        (YAMNET_PROBE_PATH)
    perch_v2_probe_path       ``None``        (PERCH_V2_PROBE_PATH)
    species_allowlist         ``None``        (FAUN_SPECIES_ALLOWLIST)
    presence_gate_k           ``0.0``         (FAUN_PRESENCE_GATE_K)
    perch_v2_calibrator_path  ``None``        (PERCH_V2_CALIBRATOR_PATH)
    prob_smoothing            ``False``       (FAUN_PROB_SMOOTHING)
    routing_enabled           ``False``       (FAUN_ROUTING_ENABLED)
    routing_tau_bird          ``0.5``         (FAUN_ROUTING_TAU_BIRD)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Documented defaults (single source of truth — referenced from from_env)
# ---------------------------------------------------------------------------

DEFAULT_JOBS_ROOT = "./jobs"
DEFAULT_CLASSIFIER = "stub"
# Source-resolution hardening limits. These MIRROR the authoritative defaults in
# faun.sources (the enforcing reader, which still reads FAUN_SOURCE_* directly).
# They are sized for the real Yandex.Disk trap folders (~23 GB extracted per A1
# folder, ~115 MB per WAV), NOT a conservative generic default — a 2 GiB cap
# would reject a legitimate trap-folder pull. Keep these two in sync until
# faun.sources is migrated to read Settings (see module docstring NOTE).
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_BYTES = 30 * 1024**3  # 30 GiB on-the-wire download cap
DEFAULT_MAX_UNCOMPRESSED_BYTES = 60 * 1024**3  # 60 GiB extracted (zip-bomb) cap
DEFAULT_MAX_ENTRIES = 100_000  # archive member cap
DEFAULT_MAX_REDIRECTS = 5  # SSRF redirect cap
DEFAULT_LOG_JSON = True

#: Routing p_bird decision threshold (FR-R1). Consulted ONLY when
#: FAUN_ROUTING_ENABLED is on. Placeholder 0.5 — calibrate with
#: scripts/calibrate_routing.py against real bird vs noise clips before enabling.
DEFAULT_ROUTING_TAU_BIRD = 0.5

#: Values accepted as boolean true / false for FAUN_LOG_JSON. Anything else
#: (including the empty string) falls back to the field default.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Defensive primitive parsers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, *, positive: bool = True) -> int:
    """Parse an int env var; fall back to ``default`` on absence/garbage.

    Args:
        name: Environment variable name.
        default: Value used when unset, empty, non-integer or (when
            ``positive``) non-positive.
        positive: When True, a parsed value <= 0 is rejected as nonsensical for
            a size/count limit and replaced by ``default``.

    Returns:
        The parsed integer, or ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("invalid int for %s=%r; using default %d", name, raw, default)
        return default
    if positive and value <= 0:
        logger.warning("non-positive %s=%d; using default %d", name, value, default)
        return default
    return value


def _env_float(name: str, default: float, *, positive: bool = True) -> float:
    """Parse a float env var; fall back to ``default`` on absence/garbage.

    See :func:`_env_int`; the same defensive semantics apply for floats (e.g.
    a network timeout in seconds).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("invalid float for %s=%r; using default %s", name, raw, default)
        return default
    if positive and value <= 0:
        logger.warning("non-positive %s=%s; using default %s", name, value, default)
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; fall back to ``default`` on absence/unknown."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default


def _env_path_opt(name: str) -> str | None:
    """Return a stripped non-empty env string, else None (blank == unset)."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _env_secret_opt(name: str) -> str | None:
    """Return an env SECRET verbatim (NO strip), or None if unset / empty string.

    Unlike :func:`_env_path_opt`, credentials are not stripped: a password may
    legitimately contain surrounding spaces, and an all-whitespace value must
    stay truthy so the auth gate fails CLOSED (stays enabled) instead of silently
    disabling itself. Only a missing or empty-string value collapses to ``None``.
    """
    raw = os.environ.get(name)
    return raw or None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable, typed snapshot of every Faun configuration knob.

    Construct via :meth:`from_env`; access the shared instance via
    :func:`get_settings`. The dataclass is frozen so configuration cannot be
    mutated after parse — callers that need a different config in a test clear
    the :func:`get_settings` cache and re-read the environment.
    """

    # Core service knobs (match today's faun.api / faun.health behaviour).
    jobs_root: Path = Path(DEFAULT_JOBS_ROOT)
    classifier: str = DEFAULT_CLASSIFIER

    # Source-resolution hardening limits (SSRF / zip-bomb; see ADR-0001).
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_bytes: int = DEFAULT_MAX_BYTES
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_redirects: int = DEFAULT_MAX_REDIRECTS

    # Model source resolution (centralizes the adapters' own env reads).
    perch_v2_model_path: str | None = None
    perch_model_path: str | None = None
    yamnet_probe_path: str | None = None
    # Trained species probe over Perch 2 embeddings (PerchProbeAdapter). Like
    # yamnet_probe_path: an operator-supplied local pickle, never network input.
    perch_v2_probe_path: str | None = None

    # Regional species allow-list (ADR-0004). Path to a checklist file (one
    # binomial per line), or the literal "default"/"reserve" sentinel for the
    # bundled reserve seed. UNSET -> the MaskedClassifier is OFF (prod output
    # byte-for-byte unchanged); set -> the served top-k predictions are filtered
    # to the listed species (an output filter, not a logit-argmax restriction).
    species_allowlist: str | None = None

    # HTTP Basic Auth (env-gated). Both must be set to enable the site-wide
    # login gate in faun.api; either unset -> auth disabled (default-open).
    basic_user: str | None = None
    basic_pass: str | None = None

    # Presence soft-gate strength (FR-003, ADR-0007). k=0 (default) is a literal
    # no-op: Perch2Adapter returns the raw logit in `probability`, unchanged. Only
    # k>0 activates the gate `clamp01(p_species·(1+p_bird·k))` over softmax probs,
    # so the served probability changes ONLY when explicitly enabled. Parsed with
    # positive=False — 0 is the valid OFF value, NOT a rejected non-positive.
    presence_gate_k: float = 0.0

    # Serve-time probability calibrator (FR-006-serve, ADR-0007). Path to a pickled
    # TemperatureCalibrator; unset -> Prediction.prob_calibrated stays None and the
    # output is unchanged. Like the probe paths: an operator-supplied local pickle.
    perch_v2_calibrator_path: str | None = None

    # Per-recording probability smoothing sidecar (FR-004, ADR-0006). OFF by
    # default: when False, run_pipeline writes no prob_smoothed.json and the job
    # directory is byte-for-byte what it was before. Output-only — the raw
    # probability, results.csv and detections.jsonl are never touched.
    prob_smoothing: bool = False

    # Two-model routing (FR-R1/FR-R2, Wave 2). OFF by default: when
    # routing_enabled is False no RoutingClassifier is built and output is
    # unchanged. routing_tau_bird is consulted ONLY when enabled; parsed with
    # positive=False since a 0.0 threshold is a valid (never-reject) config.
    routing_enabled: bool = False
    routing_tau_bird: float = DEFAULT_ROUTING_TAU_BIRD

    # Observability.
    log_json: bool = DEFAULT_LOG_JSON

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from ``os.environ`` with safe typed defaults.

        Every numeric var is parsed defensively (malformed or non-positive
        values fall back to the documented default and emit a warning), so a
        single typo in deployment env never crashes service boot. ``classifier``
        is normalized (trimmed + lower-cased) to match the existing
        ``faun.api._build_classifier`` contract.

        Returns:
            A fully-populated, frozen :class:`Settings`.
        """
        return cls(
            jobs_root=Path(os.environ.get("FAUN_JOBS_ROOT", DEFAULT_JOBS_ROOT)),
            classifier=(
                os.environ.get("FAUN_CLASSIFIER", DEFAULT_CLASSIFIER).strip().lower()
                or DEFAULT_CLASSIFIER
            ),
            timeout_s=_env_float("FAUN_SOURCE_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            max_bytes=_env_int("FAUN_SOURCE_MAX_BYTES", DEFAULT_MAX_BYTES),
            max_uncompressed_bytes=_env_int(
                "FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES
            ),
            max_entries=_env_int("FAUN_SOURCE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES),
            max_redirects=_env_int("FAUN_SOURCE_MAX_REDIRECTS", DEFAULT_MAX_REDIRECTS),
            perch_v2_model_path=_env_path_opt("PERCH_V2_MODEL_PATH"),
            perch_model_path=_env_path_opt("PERCH_MODEL_PATH"),
            yamnet_probe_path=_env_path_opt("YAMNET_PROBE_PATH"),
            perch_v2_probe_path=_env_path_opt("PERCH_V2_PROBE_PATH"),
            species_allowlist=_env_path_opt("FAUN_SPECIES_ALLOWLIST"),
            basic_user=_env_secret_opt("FAUN_BASIC_USER"),
            basic_pass=_env_secret_opt("FAUN_BASIC_PASS"),
            # positive=False: k=0 is the valid OFF value, not a rejected non-positive.
            presence_gate_k=_env_float("FAUN_PRESENCE_GATE_K", 0.0, positive=False),
            perch_v2_calibrator_path=_env_path_opt("PERCH_V2_CALIBRATOR_PATH"),
            prob_smoothing=_env_bool("FAUN_PROB_SMOOTHING", False),
            routing_enabled=_env_bool("FAUN_ROUTING_ENABLED", False),
            routing_tau_bird=_env_float(
                "FAUN_ROUTING_TAU_BIRD", DEFAULT_ROUTING_TAU_BIRD, positive=False
            ),
            log_json=_env_bool("FAUN_LOG_JSON", DEFAULT_LOG_JSON),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` singleton.

    Cached via ``lru_cache``; call ``get_settings.cache_clear()`` to invalidate
    (tests rely on this to pick up per-test environment overrides).
    """
    return Settings.from_env()
