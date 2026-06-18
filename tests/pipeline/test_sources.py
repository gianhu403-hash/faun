"""Tests for faun.sources — URL / Yandex.Disk ingest with SSRF + zip hardening.

No real network: httpx is driven through a fake transport / fake client so the
SSRF, size, zip-bomb and zip-slip gates are exercised deterministically. The
regression gate proves resolve_source never calls ``Path()`` on a URL.
"""

from __future__ import annotations

import io
import socket
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf

from faun import sources
from faun.sources import (
    SourceError,
    _assert_public_host,
    _is_url,
    _is_yandex_disk,
    resolve_source,
    source_provenance,
)

YA_ROOT = "https://disk.yandex.ru/d/BymtYnK8E92M1Q"
YA_SUB = YA_ROOT + "/A1"
GENERIC_ZIP_URL = "https://example.com/traps.zip"


# ---------------------------------------------------------------------------
# Fixture archive helpers
# ---------------------------------------------------------------------------


def _wav_bytes(seconds: float = 0.05, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    data = np.zeros(int(sr * seconds), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _flat_wav_zip(names: list[str]) -> bytes:
    """A ZIP of flat WAVs (a single trap folder's worth of recordings)."""
    buf = io.BytesIO()
    payload = _wav_bytes()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, payload)
    return buf.getvalue()


def _zip_slip_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.wav", b"pwned")
    return buf.getvalue()


def _zip_with_entries(n: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n):
            zf.writestr(f"f{i}.wav", b"x")
    return buf.getvalue()


def _zip_with_big_uncompressed(total: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.wav", b"\0" * total)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fake httpx client
# ---------------------------------------------------------------------------


class _FakeURL:
    def __init__(self, url: str) -> None:
        self._url = url

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._url

    @property
    def host(self) -> str:
        # strip scheme + path
        rest = self._url.split("://", 1)[-1]
        host = rest.split("/", 1)[0]
        return host.split("@")[-1].split(":")[0].strip("[]")


class _FakeStreamResponse:
    """Mimics the object yielded by ``client.stream("GET", url)``."""

    def __init__(self, url: str, *, status: int, body: bytes = b"") -> None:
        self.url = _FakeURL(url)
        self.status_code = status
        self._body = body
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise sources.httpx.HTTPStatusError(  # type: ignore[attr-defined]
                f"status {self.status_code}", request=None, response=None
            )

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class FakeClient:
    """Records API calls and serves canned JSON / streamed bytes."""

    def __init__(
        self,
        *,
        json_map: dict | None = None,
        download_href: str = "https://downloader.disk.yandex.ru/zip/abc",
        final_url: str = "https://zipper-external.disk.yandex.net/zip/abc",
        stream_body: bytes = b"",
        stream_status: int = 200,
    ) -> None:
        self.json_map = json_map or {}
        self.download_href = download_href
        self.final_url = final_url
        self.stream_body = stream_body
        self.stream_status = stream_status
        self.get_calls: list[tuple[str, dict]] = []
        self.stream_calls: list[str] = []

    # generic GET used for the JSON metadata API
    def get(self, url: str, *, params: dict | None = None, **kw):
        self.get_calls.append((url, params or {}))
        body = {"href": self.download_href}
        if "/resources/download" not in url:
            body = self.json_map.get("resource", {"type": "dir", "_embedded": {}})
        return _FakeJsonResponse(url, body)

    def stream(self, method: str, url: str, **kw):
        self.stream_calls.append(url)
        # the data download streams from final_url (post-redirect)
        return _FakeStreamResponse(
            self.final_url, status=self.stream_status, body=self.stream_body
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _FakeJsonResponse:
    def __init__(self, url: str, body: dict) -> None:
        self.url = _FakeURL(url)
        self.status_code = 200
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


# ---------------------------------------------------------------------------
# 1. Regression: never Path() a URL
# ---------------------------------------------------------------------------


def test_local_existing_path_passes_through_unchanged(tmp_path: Path) -> None:
    (tmp_path / "A1").mkdir()
    out = resolve_source(str(tmp_path), tmp_path / "_wd")
    assert out == Path(str(tmp_path))


def test_yandex_url_never_constructs_path_of_url(tmp_path: Path) -> None:
    client = FakeClient(stream_body=_flat_wav_zip(["a.wav", "b.wav"]))
    out = resolve_source(YA_SUB, tmp_path, client=client)
    s = str(out)
    assert "https:/" not in s
    assert "_source" in s
    assert out.is_dir()
    # the returned dir must be ingest-scannable
    from faun.ingest import scan

    manifest = scan(out)
    assert len(manifest.entries) == 2


# ---------------------------------------------------------------------------
# 2. Yandex subfolder: public_key=root + path=/A1
# ---------------------------------------------------------------------------


def test_yandex_subfolder_splits_public_key_and_path(tmp_path: Path) -> None:
    client = FakeClient(stream_body=_flat_wav_zip(["x.wav"]))
    resolve_source(YA_SUB, tmp_path, client=client)
    # find the /resources/download call params
    dl = [(u, p) for (u, p) in client.get_calls if "/resources/download" in u]
    assert dl, "expected a /resources/download API call"
    _, params = dl[0]
    assert params["public_key"] == YA_ROOT  # root, NOT folded with /A1
    assert params["path"] == "/A1"


def test_yandex_root_without_subfolder(tmp_path: Path) -> None:
    client = FakeClient(stream_body=_flat_wav_zip(["x.wav"]))
    resolve_source(YA_ROOT, tmp_path, client=client)
    dl = [(u, p) for (u, p) in client.get_calls if "/resources/download" in u]
    _, params = dl[0]
    assert params["public_key"] == YA_ROOT
    assert params.get("path", "/") == "/"


# ---------------------------------------------------------------------------
# 3. SECURITY gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.zip",
        "http://[::1]/x.zip",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_ssrf_literal_internal_ips_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(SourceError) as exc:
        resolve_source(url, tmp_path, client=FakeClient())
    assert exc.value.kind == "ssrf"


def test_ssrf_dns_resolves_to_cgnat_tailnet(tmp_path: Path, monkeypatch) -> None:
    def fake_getaddrinfo(host, *a, **kw):
        return [(socket.AF_INET, None, None, "", ("100.64.0.1", 0))]

    monkeypatch.setattr(sources.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SourceError) as exc:
        resolve_source("http://evil.example.com/x.zip", tmp_path, client=FakeClient())
    assert exc.value.kind == "ssrf"


def test_ssrf_dns_resolves_to_private_10(tmp_path: Path, monkeypatch) -> None:
    def fake_getaddrinfo(host, *a, **kw):
        return [(socket.AF_INET, None, None, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(sources.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SourceError) as exc:
        resolve_source("http://evil.example.com/x.zip", tmp_path, client=FakeClient())
    assert exc.value.kind == "ssrf"


def test_ssrf_redirect_location_to_internal_ip(tmp_path: Path) -> None:
    # the data stream's FINAL url resolves to an internal host -> reject
    client = FakeClient(final_url="http://10.0.0.5/zip/abc")
    with pytest.raises(SourceError) as exc:
        resolve_source(YA_SUB, tmp_path, client=client)
    assert exc.value.kind == "ssrf"


def test_bad_scheme_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceError) as exc:
        resolve_source("ftp://example.com/x.zip", tmp_path, client=FakeClient())
    # not a URL we accept and not a local path
    assert exc.value.kind in {"bad-scheme", "not-found"}


def test_file_scheme_rejected(tmp_path: Path, monkeypatch) -> None:
    # public host check must reject non-http(s)
    with pytest.raises(SourceError) as exc:
        _assert_public_host("file:///etc/passwd")
    assert exc.value.kind == "bad-scheme"


def test_size_cap_enforced_mid_stream(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_SOURCE_MAX_BYTES", "16")
    client = FakeClient(stream_body=b"\0" * 4096)
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "too-large"


def test_zip_bomb_entry_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_SOURCE_MAX_ENTRIES", "3")
    client = FakeClient(stream_body=_zip_with_entries(10))
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "too-large"


def test_zip_bomb_uncompressed_size(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", "64")
    client = FakeClient(stream_body=_zip_with_big_uncompressed(8192))
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "too-large"


def test_zip_slip_rejected_and_nothing_written(tmp_path: Path) -> None:
    client = FakeClient(stream_body=_zip_slip_archive())
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "zip-slip"
    # nothing escaped the workdir
    assert not (tmp_path.parent / "evil.wav").exists()
    assert not (tmp_path / "evil.wav").exists()


def test_not_an_archive_rejected(tmp_path: Path) -> None:
    client = FakeClient(stream_body=b"this is not a zip file at all")
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "not-an-archive"


def test_source_dir_removed_on_failure(tmp_path: Path) -> None:
    client = FakeClient(stream_body=b"garbage")
    with pytest.raises(SourceError):
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert not (tmp_path / "_source").exists()


def test_network_5xx_after_retries(tmp_path: Path, monkeypatch) -> None:
    # zero backoff for tests
    monkeypatch.setattr(sources, "_BACKOFF_BASE_S", 0.0)
    client = FakeClient(stream_status=503, stream_body=b"")
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_ZIP_URL, tmp_path, client=client)
    assert exc.value.kind == "network"
    # retried more than once
    assert len(client.stream_calls) > 1


# ---------------------------------------------------------------------------
# 4. provenance + helpers
# ---------------------------------------------------------------------------


def test_source_provenance_modes(tmp_path: Path) -> None:
    (tmp_path / "A1").mkdir()
    assert source_provenance(str(tmp_path)) == {
        "source": str(tmp_path),
        "mode": "local",
    }
    assert source_provenance(GENERIC_ZIP_URL)["mode"] == "http"
    assert source_provenance(YA_SUB)["mode"] == "yadisk"


def test_is_url_and_is_yandex_disk() -> None:
    assert _is_url(GENERIC_ZIP_URL)
    assert _is_url(YA_SUB)
    assert not _is_url("/tmp/local/dir")
    assert _is_yandex_disk(YA_SUB)
    assert _is_yandex_disk("https://yadi.sk/d/abc/A1")
    assert not _is_yandex_disk(GENERIC_ZIP_URL)


# ---------------------------------------------------------------------------
# 5. Generic (non-Yandex) http(s) URLs are a first-class, WORKING feature.
# ---------------------------------------------------------------------------


def _public_dns(host_to_ip: dict[str, str] | None = None):
    """Build a fake getaddrinfo resolving hosts to public (or given) IPs."""
    host_to_ip = host_to_ip or {}

    def fake_getaddrinfo(host, *a, **kw):
        ip = host_to_ip.get(host, "93.184.216.34")  # public IP by default
        return [(socket.AF_INET, None, None, "", (ip, 0))]

    return fake_getaddrinfo


GENERIC_HOST_URL = "https://files.example.com/traps.zip"


def test_generic_url_success_no_ssrf_rejection(tmp_path: Path, monkeypatch) -> None:
    # Headline fix: a generic user-supplied URL whose final host is a non-Yandex
    # public host must resolve, NOT be allowlist-rejected.
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = FakeClient(
        final_url=GENERIC_HOST_URL,
        stream_body=_flat_wav_zip(["a.wav", "b.wav"]),
    )
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert out.is_dir()
    from faun.ingest import scan

    manifest = scan(out)
    assert len(manifest.entries) == 2


def test_generic_url_to_internal_host_still_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sources.socket,
        "getaddrinfo",
        _public_dns({"files.example.com": "100.64.0.1"}),
    )
    client = FakeClient(
        final_url=GENERIC_HOST_URL, stream_body=_flat_wav_zip(["a.wav"])
    )
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert exc.value.kind == "ssrf"


def test_generic_url_to_loopback_still_blocked(tmp_path: Path) -> None:
    with pytest.raises(SourceError) as exc:
        resolve_source("http://127.0.0.1/traps.zip", tmp_path, client=FakeClient())
    assert exc.value.kind == "ssrf"


# ---------------------------------------------------------------------------
# 6. Manual redirect following: every hop validated BEFORE the request.
# ---------------------------------------------------------------------------


class _RedirectClient:
    """Returns a 3xx (Location) first, then a 200 with the body.

    Records every URL passed to ``stream`` so a test can assert the internal hop
    was never actually requested.
    """

    def __init__(self, *, location: str, body: bytes, redirect_status: int = 302):
        self.location = location
        self.body = body
        self.redirect_status = redirect_status
        self.stream_calls: list[str] = []
        self._served_redirect = False

    def stream(self, method: str, url: str, **kw):
        self.stream_calls.append(url)
        if not self._served_redirect:
            self._served_redirect = True
            resp = _FakeStreamResponse(url, status=self.redirect_status, body=b"")
            resp.headers = {"Location": self.location}
            return resp
        return _FakeStreamResponse(url, status=200, body=self.body)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_redirect_hop_to_internal_ip_blocked_before_request(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = _RedirectClient(
        location="http://169.254.169.254/latest/meta-data",
        body=_flat_wav_zip(["a.wav"]),
    )
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert exc.value.kind == "ssrf"
    # the redirect was served, but the internal hop GET must NOT have been issued
    assert client.stream_calls == [GENERIC_HOST_URL]
    assert "169.254.169.254" not in " ".join(client.stream_calls)


def test_redirect_to_public_host_followed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = _RedirectClient(
        location="https://cdn.example.org/real.zip",
        body=_flat_wav_zip(["a.wav", "b.wav"]),
    )
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert out.is_dir()
    # both hops issued: original + redirect target
    assert client.stream_calls == [
        GENERIC_HOST_URL,
        "https://cdn.example.org/real.zip",
    ]


# ---------------------------------------------------------------------------
# 7. IPv4-mapped IPv6 must be unwrapped before the block check.
# ---------------------------------------------------------------------------


def test_ipv4_mapped_ipv6_cgnat_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sources.socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET6, None, None, "", ("::ffff:100.64.0.1", 0))],
    )
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_HOST_URL, tmp_path, client=FakeClient())
    assert exc.value.kind == "ssrf"


# ---------------------------------------------------------------------------
# 8. Zip-bomb enforced DURING extraction (declared size lies).
# ---------------------------------------------------------------------------


def _zip_compressible(real_bytes: int) -> bytes:
    """A real, valid, highly compressible member (inflates to ``real_bytes``)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.wav", b"\0" * real_bytes)
    return buf.getvalue()


def test_zip_bomb_cheap_precheck_rejects(tmp_path: Path, monkeypatch) -> None:
    # End-to-end: a real oversized member is rejected too-large by _extract.
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    monkeypatch.setenv("FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", "1024")
    blob = _zip_compressible(64 * 1024)
    client = FakeClient(final_url=GENERIC_HOST_URL, stream_body=blob)
    with pytest.raises(SourceError) as exc:
        resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert exc.value.kind == "too-large"


def test_zip_bomb_enforced_during_extraction(tmp_path: Path, monkeypatch) -> None:
    # Authoritative cap = streamed-bytes counter, NOT declared file_size.
    # Simulate an attacker who under-declares: the cheap pre-check sees a large cap
    # (passes), but the running streamed counter must still abort. We drive that by
    # making _int_env return a generous cap on the pre-check read and a tiny cap on
    # the streamed read, proving line 440's branch is the load-bearing control.
    big_zip = tmp_path / "big.zip"
    big_zip.write_bytes(_zip_compressible(64 * 1024))
    dest = tmp_path / "out"

    real_int_env = sources._int_env
    calls = {"n": 0}

    def staged_int_env(name, default):
        if name == "FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES":
            calls["n"] += 1
            # 1st read = streamed cap (tiny), 2nd read = cheap pre-check (generous)
            return 1024 if calls["n"] == 1 else 10 * 1024 * 1024
        return real_int_env(name, default)

    monkeypatch.setattr(sources, "_int_env", staged_int_env)
    with pytest.raises(SourceError) as exc:
        sources._extract(big_zip, dest)
    assert exc.value.kind == "too-large"
    assert calls["n"] >= 2, "streamed cap path must be reached after the pre-check"


# ---------------------------------------------------------------------------
# 9. download.zip removed after a successful extract.
# ---------------------------------------------------------------------------


def test_download_archive_removed_after_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = FakeClient(
        final_url=GENERIC_HOST_URL, stream_body=_flat_wav_zip(["a.wav"])
    )
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert out.is_dir()
    # extracted tree present; the archive file is gone.
    archives = list((tmp_path / "_source").rglob("download.zip"))
    assert archives == []
    wavs = list((tmp_path / "_source").rglob("*.wav"))
    assert wavs, "extracted tree must remain"


# ---------------------------------------------------------------------------
# 10. _single_root nested-single-dir unwrap.
# ---------------------------------------------------------------------------


def _nested_traps_zip() -> bytes:
    buf = io.BytesIO()
    payload = _wav_bytes()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("traps/A1/rec1.wav", payload)
        zf.writestr("traps/A2/rec2.wav", payload)
    return buf.getvalue()


def _nested_traps_with_stray_file_zip() -> bytes:
    buf = io.BytesIO()
    payload = _wav_bytes()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("traps/A1/rec1.wav", payload)
        zf.writestr("README.txt", b"hi")
    return buf.getvalue()


def test_single_root_unwraps_nested_traps_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = FakeClient(final_url=GENERIC_HOST_URL, stream_body=_nested_traps_zip())
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert out.name == "traps"
    from faun.ingest import scan

    manifest = scan(out)
    trap_ids = {e.trap_id for e in manifest.entries}
    assert {"A1", "A2"} <= trap_ids


def test_single_root_does_not_unwrap_with_stray_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = FakeClient(
        final_url=GENERIC_HOST_URL,
        stream_body=_nested_traps_with_stray_file_zip(),
    )
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    # a dir + a stray top-level file -> NOT unwrapped (root kept).
    assert out.name != "traps"
    assert (out / "traps").is_dir()
    assert (out / "README.txt").is_file()


# ---------------------------------------------------------------------------
# 11. Retry then RECOVER (503 then 200).
# ---------------------------------------------------------------------------


class _FlakyClient:
    """Streams a 503 on the first call, then a 200 with the body."""

    def __init__(self, *, final_url: str, body: bytes):
        self.final_url = final_url
        self.body = body
        self.stream_calls: list[str] = []
        self._n = 0

    def stream(self, method: str, url: str, **kw):
        self.stream_calls.append(url)
        self._n += 1
        status = 503 if self._n == 1 else 200
        body = b"" if status == 503 else self.body
        return _FakeStreamResponse(self.final_url, status=status, body=body)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_retry_then_recover(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources, "_BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = _FlakyClient(final_url=GENERIC_HOST_URL, body=_flat_wav_zip(["a.wav"]))
    out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    assert out.is_dir()
    assert len(client.stream_calls) >= 2


# ---------------------------------------------------------------------------
# 12. Absolute-path zip member must not escape dest.
# ---------------------------------------------------------------------------


def _abs_path_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("/tmp/pwned.wav", b"pwned")
    return buf.getvalue()


def test_absolute_path_member_does_not_escape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources.socket, "getaddrinfo", _public_dns())
    client = FakeClient(final_url=GENERIC_HOST_URL, stream_body=_abs_path_zip())
    sentinel = Path("/tmp/pwned.wav")  # noqa: S108 - test sentinel only
    existed_before = sentinel.exists()
    try:
        out = resolve_source(GENERIC_HOST_URL, tmp_path, client=client)
    except SourceError as exc:
        # blocked outright is acceptable (zip-slip / similar)
        assert exc.kind in {"zip-slip", "not-an-archive", "too-large"}
        assert not (sentinel.exists() and not existed_before)
        return
    # if it did extract, the member must land strictly inside dest, never at /tmp.
    landed = list(Path(out).rglob("pwned.wav"))
    assert landed, "member must be confined inside dest"
    assert not (sentinel.exists() and not existed_before)
