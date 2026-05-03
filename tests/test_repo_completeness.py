"""Tests for FAUN-38a: loud failure when YDB repo is missing required methods.

Background
----------
``cloud/db/incidents.py`` selects an incident-storage backend at import time:
either the in-memory implementation (default / tests) or
:class:`cloud.db.ydb_incidents.YDBIncidentRepository` when ``YDB_ENDPOINT`` is
set.  The current code uses ``getattr(_repo, "get_stale_incidents",
get_stale_incidents)`` for two methods that are NOT declared on the
:class:`cloud.db.base.IncidentRepository` ABC -- ``get_stale_incidents`` and
``get_recent_nearby_incident``.  This silent fallback is dangerous: when
``YDB_ENDPOINT`` is set but ``YDBIncidentRepository`` does not implement the
method, the module-level in-memory function is bound instead.  That function
operates on the empty ``_incidents = {}`` dict, so the cleanup-job and the
spatial-dedup logic both return ``[]`` / ``None`` silently, breaking workflow
with no visible error.

These tests are written BEFORE the Green-phase fix and are expected to FAIL
right now.  Once ``incidents.py`` switches to explicit attribute access
(``_repo.get_stale_incidents``) and ``YDBIncidentRepository`` gains the two
missing methods, all four tests should pass.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# Methods every concrete IncidentRepository must expose for the cloud module
# to be safe.  This is a SUPERSET of cloud/db/base.py's ABC because the ABC
# does not currently include get_stale_incidents / get_recent_nearby_incident
# (the contract for those is informal, which is exactly the hole this test
# closes).
REQUIRED_INCIDENT_REPO_METHODS: frozenset[str] = frozenset(
    {
        "create_incident",
        "update_status",
        "update_incident",
        "get_incident",
        "get_all_incidents",
        "get_active_incident_for_chat",
        "assign_chat_to_incident",
        "clear_chat_incident",
        # Methods currently MISSING from YDBIncidentRepository -- the silent
        # getattr-fallback in incidents.py hides this hole.
        "get_stale_incidents",
        "get_recent_nearby_incident",
    }
)


def _reload_incidents_module():
    """Re-import ``cloud.db.incidents`` so the YDB branch is re-evaluated.

    Returns the freshly-loaded module object.
    """
    import cloud.db.incidents as inc_mod

    return importlib.reload(inc_mod)


@pytest.fixture
def restore_incidents_module():
    """Restore ``cloud.db.incidents`` module state after a YDB reload test.

    Tests in this file flip ``YDB_ENDPOINT`` and reload the module to
    exercise the YDB branch.  Other test files import names directly from
    ``cloud.db.incidents`` (e.g. ``from cloud.db.incidents import
    _incidents, create_incident``); those bindings point at the ORIGINAL
    objects loaded the first time the module was imported.  A naive
    ``importlib.reload`` rebinds the in-module names to fresh objects,
    breaking the older bindings held by other test files.

    To keep neighbour tests green, this fixture:

    1. Captures the pre-test module dict (objects + bindings).
    2. Lets the test reload / mutate ``cloud.db.incidents`` freely.
    3. After the test, force-reloads the module with a clean (no-YDB)
       environment, then **copies the pre-test bindings back over** so
       other modules' ``from cloud.db.incidents import X`` references
       point at the same objects they did originally.
    """
    import cloud.db.incidents as inc_mod

    pre_test_attrs = dict(inc_mod.__dict__)
    pre_test_incidents = inc_mod._incidents
    pre_test_chat_map = inc_mod._chat_to_incident

    yield

    # Drop cached YDB submodules so the reload below is clean.
    for mod_name in ("cloud.db.ydb_incidents", "cloud.db.ydb_client"):
        sys.modules.pop(mod_name, None)
    # Reload the in-memory variant (env is already restored by monkeypatch).
    importlib.reload(inc_mod)
    # Restore the original module attributes so test files that did
    # ``from cloud.db.incidents import X`` continue to see object X.
    inc_mod.__dict__.update(pre_test_attrs)
    # And clear the original mutable state so other tests see empty dicts.
    pre_test_incidents.clear()
    pre_test_chat_map.clear()


# ---------------------------------------------------------------------------
# Test 1 -- class-level interface check (no instantiation, no env)
# ---------------------------------------------------------------------------


def test_ydb_incident_repo_implements_required_methods():
    """``YDBIncidentRepository`` must expose every required method.

    Today this fails because ``get_stale_incidents`` and
    ``get_recent_nearby_incident`` are not implemented on the YDB class.
    """
    from cloud.db.ydb_incidents import YDBIncidentRepository

    missing = sorted(
        name
        for name in REQUIRED_INCIDENT_REPO_METHODS
        if not hasattr(YDBIncidentRepository, name)
    )
    assert not missing, (
        "YDBIncidentRepository is missing required methods: "
        f"{missing}.  cloud/db/incidents.py currently masks this with a "
        "silent getattr fallback to the in-memory implementation, which "
        "operates on an empty dict in YDB mode and silently breaks the "
        "cleanup-job and spatial dedup."
    )


# ---------------------------------------------------------------------------
# Test 2 -- module dispatches to YDB-bound method when YDB is active
# ---------------------------------------------------------------------------


def test_module_dispatches_get_stale_incidents_to_ydb_when_active(
    monkeypatch, restore_incidents_module
):
    """When ``YDB_ENDPOINT`` is set, ``incidents.get_stale_incidents`` must
    be the YDB repo's bound method, not the module-level in-memory function.

    Today this fails because the ``getattr(_repo, "get_stale_incidents",
    get_stale_incidents)`` fallback returns the in-memory function whose
    ``__qualname__`` is ``"get_stale_incidents"`` (no ``"YDBIncidentRepository."``
    prefix).  After the Green fix the YDB repo implements the method and
    ``incidents.py`` binds it explicitly, so the qualname check passes.
    """
    monkeypatch.setenv("YDB_ENDPOINT", "fake://test-endpoint")

    # Stub out network/IO entry points so YDBIncidentRepository.__init__
    # does not try to talk to a real YDB cluster.
    from cloud.db import ydb_client

    monkeypatch.setattr(ydb_client, "ensure_tables", lambda: None)
    monkeypatch.setattr(ydb_client, "get_pool", lambda: None)

    inc_mod = _reload_incidents_module()

    qualname = getattr(inc_mod.get_stale_incidents, "__qualname__", "")
    assert qualname.startswith("YDBIncidentRepository."), (
        "incidents.get_stale_incidents is bound to the in-memory function "
        f"(qualname={qualname!r}) when YDB_ENDPOINT is set.  This means the "
        "silent getattr fallback is masking a missing YDB method -- the "
        "in-memory function would scan an empty dict and silently report "
        "no stale incidents."
    )


# ---------------------------------------------------------------------------
# Test 3 -- importing with an incomplete YDB repo must fail LOUDLY
# ---------------------------------------------------------------------------


def test_module_raises_loudly_if_ydb_repo_missing_method(
    monkeypatch, restore_incidents_module
):
    """Importing ``cloud.db.incidents`` with ``YDB_ENDPOINT`` set and a
    YDB repo class that lacks ``get_stale_incidents`` MUST raise
    ``AttributeError`` -- not silently fall back to the in-memory function.

    Today this fails: the reload completes successfully because
    ``getattr(..., default)`` swallows the missing-method case.
    """
    monkeypatch.setenv("YDB_ENDPOINT", "fake://test-endpoint")

    # Build a deliberately incomplete repo: it can be instantiated and
    # implements most methods, but is missing get_stale_incidents.
    class IncompleteYDBRepo:
        def __init__(self) -> None:  # no network, no IO
            return

        def create_incident(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def get_incident(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def get_active_incident_for_chat(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def assign_chat_to_incident(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def clear_chat_incident(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def update_status(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def update_incident(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return None

        def get_all_incidents(self, *args, **kwargs):  # noqa: ANN001, ANN002
            return []

        # Note: get_stale_incidents and get_recent_nearby_incident are
        # intentionally absent.

    # Patch the YDB repo class at the source module so the freshly-imported
    # ``cloud.db.incidents`` resolves the patched class.
    import cloud.db.ydb_incidents as ydb_inc_mod

    monkeypatch.setattr(ydb_inc_mod, "YDBIncidentRepository", IncompleteYDBRepo)

    # Stub out the YDB client surface so accidental calls do not hit the network.
    from cloud.db import ydb_client

    monkeypatch.setattr(ydb_client, "ensure_tables", lambda: None)
    monkeypatch.setattr(ydb_client, "get_pool", lambda: None)

    with pytest.raises(AttributeError, match="get_stale_incidents"):
        _reload_incidents_module()


# ---------------------------------------------------------------------------
# Test 4 -- instance-level binding check (callable on a real instance)
# ---------------------------------------------------------------------------


def test_ydb_repo_methods_callable_on_instance(monkeypatch):
    """A live ``YDBIncidentRepository`` instance must expose
    ``get_stale_incidents`` and ``get_recent_nearby_incident`` as callables.

    Today this fails with ``AttributeError`` because the methods do not
    exist on the class (and thus not on instances either).  Mostly redundant
    with test 1 but explicitly probes instance binding -- after the Green
    fix the YDB repo implements both methods, so attribute lookup on an
    instance succeeds and they are callable.
    """
    # Stub out the YDB client so __init__ does not hit the network.
    from cloud.db import ydb_client

    monkeypatch.setattr(ydb_client, "ensure_tables", lambda: None)
    monkeypatch.setattr(ydb_client, "get_pool", lambda: None)

    from cloud.db.ydb_incidents import YDBIncidentRepository

    repo = YDBIncidentRepository()

    assert callable(getattr(repo, "get_stale_incidents", None)), (
        "YDBIncidentRepository instance has no callable get_stale_incidents "
        "-- cleanup-job will silently scan an empty in-memory dict in YDB mode."
    )
    assert callable(getattr(repo, "get_recent_nearby_incident", None)), (
        "YDBIncidentRepository instance has no callable "
        "get_recent_nearby_incident -- spatial dedup will silently allow "
        "duplicate incidents in YDB mode."
    )
