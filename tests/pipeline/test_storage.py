"""Tests for faun.storage — Storage protocol + LocalFSStorage.

Covers put/get round-trip, url() forms (file:// and relative), key sandboxing
(absolute / traversal rejection), missing-file errors, and Protocol conformance.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

from faun.storage import LocalFSStorage, Storage


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_localfs_is_storage(tmp_path: Path) -> None:
    store = LocalFSStorage(tmp_path)
    assert isinstance(store, Storage)


def test_root_created_on_init(tmp_path: Path) -> None:
    root = tmp_path / "store-root"
    store = LocalFSStorage(root)
    assert store.root.is_dir()


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_put_then_get(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        src = tmp_path / "src.txt"
        src.write_text("hello faun", encoding="utf-8")

        store.put(src, "job-1/results.csv")
        dst = tmp_path / "out" / "copy.csv"
        store.get("job-1/results.csv", dst)
        assert dst.read_text(encoding="utf-8") == "hello faun"

    def test_put_creates_nested_key_dirs(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        src = tmp_path / "src.bin"
        src.write_bytes(b"\x00\x01\x02")
        store.put(src, "a/b/c/d.bin")
        assert (store.root / "a" / "b" / "c" / "d.bin").read_bytes() == b"\x00\x01\x02"

    def test_put_missing_source_raises(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            store.put(tmp_path / "nope.txt", "k")

    def test_get_missing_key_raises(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            store.get("missing/key.txt", tmp_path / "out.txt")


# ---------------------------------------------------------------------------
# url()
# ---------------------------------------------------------------------------


class TestUrl:
    def test_url_is_file_uri_by_default(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        url = store.url("job-1/results.csv")
        assert url.startswith("file://")
        # The URI must resolve back to the on-disk path under root.
        parsed = urlparse(url)
        resolved = Path(url2pathname(parsed.path))
        assert resolved == store.root / "job-1" / "results.csv"

    def test_url_relative_mode(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store", relative_urls=True)
        assert store.url("job-1/results.csv") == "job-1/results.csv"

    def test_url_is_pure_mapping_no_existence_required(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        # Object does not exist yet; url() should still return a locator.
        assert store.url("never/created.csv").startswith("file://")


# ---------------------------------------------------------------------------
# Key sandboxing
# ---------------------------------------------------------------------------


class TestKeySandbox:
    @pytest.mark.parametrize("bad_key", ["", "   "])
    def test_empty_key_rejected(self, tmp_path: Path, bad_key: str) -> None:
        store = LocalFSStorage(tmp_path / "store")
        with pytest.raises(ValueError):
            store.url(bad_key)

    def test_absolute_key_rejected(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        with pytest.raises(ValueError):
            store.url("/etc/passwd")

    @pytest.mark.parametrize(
        "bad_key",
        ["../escape.txt", "a/../../escape.txt", "..", "ok/../../x"],
    )
    def test_traversal_rejected(self, tmp_path: Path, bad_key: str) -> None:
        store = LocalFSStorage(tmp_path / "store")
        with pytest.raises(ValueError):
            store.url(bad_key)

    def test_traversal_rejected_on_put(self, tmp_path: Path) -> None:
        store = LocalFSStorage(tmp_path / "store")
        src = tmp_path / "src.txt"
        src.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            store.put(src, "../escape.txt")

    def test_dot_inside_key_is_fine(self, tmp_path: Path) -> None:
        # A single dot or normal filenames with dots must NOT be rejected.
        store = LocalFSStorage(tmp_path / "store")
        src = tmp_path / "src.txt"
        src.write_text("x", encoding="utf-8")
        store.put(src, "job.1/results.final.csv")
        assert (store.root / "job.1" / "results.final.csv").is_file()
