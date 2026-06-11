"""Storage: ``Storage`` protocol + ``LocalFSStorage`` (Phase 2).

S3-backed storage is a July task and is intentionally NOT implemented here —
``LocalFSStorage`` is the single concrete backend for now.

A *key* is a forward-slash-separated relative path within the storage root
(e.g. ``"job-42/results.csv"``). Keys are sandboxed to ``root``: attempts to
escape it (absolute keys, ``..`` traversal) raise ``ValueError``.

stdlib only — no heavy imports here.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

__all__ = ["Storage", "LocalFSStorage"]


@runtime_checkable
class Storage(Protocol):
    """Frozen object-store interface (see faun/INTERFACES.md).

    ``key`` is an opaque, backend-relative identifier. Implementations must not
    let a key escape their namespace.
    """

    def put(self, local_path: str | Path, key: str) -> None:
        """Upload/copy the file at ``local_path`` to ``key``."""
        ...

    def get(self, key: str, local_path: str | Path) -> None:
        """Download/copy the object at ``key`` to ``local_path``."""
        ...

    def url(self, key: str) -> str:
        """Return a locator for ``key`` (scheme depends on the backend)."""
        ...


def _safe_relative(key: str) -> PurePosixPath:
    """Validate ``key`` and return it as a sandbox-safe relative posix path.

    Raises ``ValueError`` for empty keys, absolute keys, or any key that would
    traverse outside the root via ``..``.
    """
    if not key or not key.strip():
        raise ValueError("storage key must be a non-empty string")
    rel = PurePosixPath(key)
    if rel.is_absolute():
        raise ValueError(f"storage key must be relative, got {key!r}")
    parts = rel.parts
    if any(part == ".." for part in parts):
        raise ValueError(f"storage key must not contain '..': {key!r}")
    return rel


class LocalFSStorage:
    """Filesystem-backed :class:`Storage` rooted at ``root``.

    Files are stored at ``root / key``. ``url(key)`` returns a ``file://`` URI
    pointing at the absolute on-disk location; pass ``relative_urls=True`` to
    instead return the plain key (a path relative to ``root``).
    """

    def __init__(self, root: str | Path, *, relative_urls: bool = False) -> None:
        self._root = Path(root).expanduser().resolve()
        self._relative_urls = relative_urls
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str) -> Path:
        rel = _safe_relative(key)
        return self._root.joinpath(*rel.parts)

    def put(self, local_path: str | Path, key: str) -> None:
        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(f"source file not found: {src}")
        dst = self._resolve(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def get(self, key: str, local_path: str | Path) -> None:
        src = self._resolve(key)
        if not src.is_file():
            raise FileNotFoundError(f"object not found for key {key!r}: {src}")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def url(self, key: str) -> str:
        """Return a locator for ``key``.

        Default: a ``file://`` URI for the absolute path under ``root``.
        With ``relative_urls=True``: the key itself (relative to ``root``).
        Existence is not required — ``url`` is a pure mapping.
        """
        rel = _safe_relative(key)
        if self._relative_urls:
            return rel.as_posix()
        return self._resolve(key).as_uri()
