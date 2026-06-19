"""faun.sources — resolve a *source* (local path / URL / Yandex.Disk) to a
local directory that :func:`faun.ingest.scan` can read.

P0 fix. The pipeline used to do ``Path("https://disk.yandex.ru/d/...")`` which
collapses ``//`` and silently produces a bogus relative path; ingest then fails.
:func:`resolve_source` instead:

* passes an **existing local path** straight through (never ``Path()`` on a URL);
* for a generic ``http(s)`` archive URL or a Yandex.Disk share, downloads + safely
  extracts an archive under ``workdir/_source/`` and returns the local extraction
  dir. Generic user-supplied URLs are first-class: the user chose the host, so the
  SSRF IP guard on the (manually followed) redirect chain is the control — there is
  **no** host allowlist on the generic path.

Security is merge-blocking:

* **SSRF** — every target host (initial + *every* redirect hop) is resolved to
  IP(s) via ``socket.getaddrinfo`` and rejected if any IP is private / loopback /
  link-local / reserved / CGNAT (``100.64.0.0/10`` is the cluster tailnet).
  Redirects are followed **manually** (``follow_redirects=False``): each hop's
  ``Location`` is validated *before* the next request is issued, so we never touch
  an internal host even once.
* **Allowlist (Yandex path only)** — a Yandex.Disk download href comes from the
  Yandex API and could redirect anywhere, so for that path the final/redirect
  hosts must additionally be in the Yandex data-host allowlist. For a generic URL
  the user owns the host, so no allowlist is applied. Because DNS-rebind TOCTOU
  (``getaddrinfo`` vs the socket httpx actually connects) is a known residual, the
  IP guard is **best-effort**; for the Yandex path the host allowlist is the
  load-bearing control.
* **Size** — enforced *during* streaming; the download aborts mid-stream once the
  cap is exceeded.
* **Zip-bomb** — entry count is bounded and the uncompressed cap is enforced
  *during* extraction by a running counter of bytes actually written (declared
  ``file_size`` is only a cheap fast-reject, never the authoritative cap).
* **Zip-slip** — any member resolving outside the destination dir is refused and
  nothing is written outside ``dest``.
* **Cleanup** — ``workdir/_source/`` is removed on any failure; on success the
  downloaded archive is deleted immediately (only the unpacked tree is kept).

Heavy import (``httpx``) is module-level here because httpx is a core pipeline
dependency (requirements-pipeline.txt); TF-class deps are not pulled in.

Configuration: the SSRF / zip-bomb / size / timeout limits are sourced from
:func:`faun.settings.get_settings` (the single typed home for ``FAUN_SOURCE_*``),
read at call time so a per-test environment override picks up after
``get_settings.cache_clear()``. The ``_int_env`` helper remains the call-time
indirection for the uncompressed cap (its default now comes from Settings).
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import zipfile
from pathlib import Path
from typing import Any

import httpx

from faun.settings import get_settings

__all__ = [
    "SourceError",
    "resolve_source",
    "source_provenance",
]

# ---------------------------------------------------------------------------
# Configuration — limits live in faun.settings (FAUN_SOURCE_* knobs). Retry
# tuning is local: tests patch _MAX_RETRIES / _BACKOFF_BASE_S directly.
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5  # patched to 0 in tests

# Yandex.Disk public API + the exact data hosts its download hrefs use.
_YADISK_API = "https://cloud-api.yandex.net/v1/disk/public"
_YADISK_LINK_RE = re.compile(
    r"^(https?://(?:disk[.]yandex[.][a-z]+|yadi[.]sk)/[di]/[^/?#]+)(/[^?#]*)?",
    re.IGNORECASE,
)
# Hosts we permit data to come from (exact + suffix). SSRF still IP-checks them.
_ALLOWED_DATA_HOSTS_EXACT = frozenset(
    {
        "cloud-api.yandex.net",
        "downloader.disk.yandex.ru",
        "zipper-external.disk.yandex.net",
    }
)
_ALLOWED_DATA_HOST_SUFFIXES = (".storage.yandex.net", ".disk.yandex.net")


class SourceError(RuntimeError):
    """A source could not be resolved.

    ``kind`` is a stable taxonomy tag in::

        {"bad-scheme", "ssrf", "not-found", "network", "too-large",
         "zip-slip", "not-an-archive", "empty"}
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Settings access (cheap; re-read so tests can monkeypatch env)
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    """Read an int ``FAUN_SOURCE_*`` env var, falling back to ``default``.

    Call-time indirection for the uncompressed cap in :func:`_extract`: the
    ``default`` is supplied by ``get_settings()`` (which itself reflects the env
    after a cache clear), while a direct env override still wins here so a
    per-test ``monkeypatch.setenv`` takes effect without clearing the cache.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def _is_url(src: str) -> bool:
    """True for an ``http://`` / ``https://`` source string."""
    return src.lower().startswith(("http://", "https://"))


def _is_yandex_disk(src: str) -> bool:
    """True for a public Yandex.Disk share link (``disk.yandex.*`` or ``yadi.sk``)."""
    return bool(_YADISK_LINK_RE.match(src.strip()))


# ---------------------------------------------------------------------------
# SSRF
# ---------------------------------------------------------------------------


def _host_of(url: str) -> str:
    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0]
    host = host.split("@")[-1]  # drop any userinfo
    # strip :port — but keep IPv6 brackets handling
    if host.startswith("["):
        return host[1 : host.index("]")] if "]" in host else host.strip("[]")
    return host.split(":")[0]


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    """Reject loopback / private / link-local / reserved / CGNAT addresses."""
    # An IPv4-mapped IPv6 (``::ffff:100.64.0.1``) must be judged on its embedded
    # IPv4: stdlib flags it neither private nor CGNAT in the v6 form.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    # CGNAT 100.64.0.0/10 — the cluster tailnet. Not flagged is_private by stdlib.
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network(
        "100.64.0.0/10"
    ):
        return True
    # Belt-and-suspenders: refuse any non-global address that slipped the above.
    if not ip.is_global:
        return True
    return False


def _assert_public_host(url: str) -> str:
    """Validate the scheme + resolve the host and reject internal targets.

    Returns the host on success. Raises :class:`SourceError`
    (``kind="bad-scheme"`` or ``kind="ssrf"``).
    """
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme not in ("http", "https"):
        raise SourceError(f"refusing non-http(s) scheme: {url!r}", kind="bad-scheme")

    host = _host_of(url)
    if not host:
        raise SourceError(f"no host in URL: {url!r}", kind="ssrf")

    # Literal IP host -> check directly.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_is_blocked(literal):
            raise SourceError(f"refusing internal IP host: {host}", kind="ssrf")
        return host

    # Hostname -> resolve every address and reject if ANY is internal.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:  # pragma: no cover - DNS failure path
        raise SourceError(f"cannot resolve host {host!r}: {exc}", kind="network")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:  # pragma: no cover - defensive
            continue
        if _ip_is_blocked(ip):
            raise SourceError(
                f"host {host!r} resolves to internal IP {addr}", kind="ssrf"
            )
    return host


def _host_allowlisted(host: str) -> bool:
    if host in _ALLOWED_DATA_HOSTS_EXACT:
        return True
    return any(host.endswith(suf) for suf in _ALLOWED_DATA_HOST_SUFFIXES)


# ---------------------------------------------------------------------------
# Yandex.Disk resolution
# ---------------------------------------------------------------------------


def _split_yadisk(src: str) -> tuple[str, str]:
    """Split a share link into ``(public_key, subpath)``.

    The subfolder MUST NOT be folded into ``public_key``: the share root is the
    public_key, the subfolder goes in a separate ``&path=``. Returns ``"/"`` for
    a bare root link.
    """
    m = _YADISK_LINK_RE.match(src.strip())
    if m is None:  # pragma: no cover - guarded by caller
        raise SourceError(f"not a Yandex.Disk link: {src!r}", kind="bad-scheme")
    public_key = m.group(1)
    subpath = m.group(2) or "/"
    return public_key, subpath


def _resolve_yandex_href(src: str, client: httpx.Client) -> str:
    """Resolve the direct download href for a Yandex.Disk (public_key, subpath).

    For a folder the href yields a ZIP. Raises ``SourceError(kind="not-found")``
    when the API reports the resource is missing.
    """
    public_key, subpath = _split_yadisk(src)
    params = {"public_key": public_key, "path": subpath}
    url = f"{_YADISK_API}/resources/download"
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - exercised via fakes
        raise SourceError(f"Yandex.Disk API error: {exc}", kind="not-found")
    except httpx.HTTPError as exc:  # pragma: no cover
        raise SourceError(f"Yandex.Disk network error: {exc}", kind="network")
    href = resp.json().get("href")
    if not href:
        raise SourceError("Yandex.Disk returned no download href", kind="not-found")
    return href


# ---------------------------------------------------------------------------
# Download (streaming, size-bounded, SSRF re-check on final URL)
# ---------------------------------------------------------------------------


def _validate_hop(url: str, *, is_yandex: bool) -> None:
    """SSRF-validate one URL we are about to request.

    Always enforces the IP guard. For the Yandex path the host must additionally
    be allowlisted (the href came from the Yandex API and could redirect anywhere).
    """
    host = _assert_public_host(url)
    if is_yandex and not _host_allowlisted(host):
        raise SourceError(f"download host not allowlisted: {host}", kind="ssrf")


def _stream_to_file(
    href: str, dest_file: Path, client: httpx.Client, *, is_yandex: bool
) -> None:
    """Follow redirects MANUALLY and stream the final 2xx body to ``dest_file``.

    Every hop (initial href + each ``Location``) is SSRF-validated *before* the
    request that targets it is issued, bounded by ``FAUN_SOURCE_MAX_REDIRECTS``.
    Raises ``SourceError`` (``network`` is retryable upstream).
    """
    settings = get_settings()
    max_bytes = settings.max_bytes
    max_redirects = settings.max_redirects

    url = href
    for _ in range(max_redirects + 1):
        # Validate the host we are ABOUT to hit, before issuing the request.
        _validate_hop(url, is_yandex=is_yandex)
        with client.stream("GET", url) as resp:
            status = getattr(resp, "status_code", 200)
            if 300 <= status < 400:
                location = resp.headers.get("Location") or resp.headers.get("location")
                if not location:
                    raise SourceError(
                        f"redirect {status} without Location", kind="network"
                    )
                # Resolve a relative Location against the current absolute URL.
                url = str(httpx.URL(url).join(location))
                continue
            if status >= 500:
                raise SourceError(f"upstream {status}", kind="network")
            if status >= 400:
                raise SourceError(f"upstream {status}", kind="not-found")

            # Defence in depth: if the transport still surfaced a different final
            # URL (a redirect we did not drive), validate it before reading body.
            final_url = str(getattr(resp, "url", url))
            if final_url and final_url != url:
                _validate_hop(final_url, is_yandex=is_yandex)

            written = 0
            with open(dest_file, "wb") as fh:
                for chunk in resp.iter_bytes():
                    written += len(chunk)
                    if written > max_bytes:
                        raise SourceError(
                            f"download exceeds {max_bytes} bytes", kind="too-large"
                        )
                    fh.write(chunk)
            if written == 0:
                raise SourceError("downloaded empty body", kind="empty")
            return

    raise SourceError(f"too many redirects (> {max_redirects})", kind="network")


def _download(
    href: str, dest_file: Path, client: httpx.Client, *, is_yandex: bool
) -> None:
    """Stream ``href`` to ``dest_file`` with size enforcement + SSRF re-check.

    Bounded exponential backoff retries on 5xx / timeout. Aborts mid-stream once
    ``FAUN_SOURCE_MAX_BYTES`` is exceeded. Redirects are followed manually so that
    every hop is SSRF-checked before being requested (see :func:`_stream_to_file`).
    """
    import time

    last_err: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            _stream_to_file(href, dest_file, client, is_yandex=is_yandex)
            return
        except SourceError as exc:
            if exc.kind == "network" and attempt < _MAX_RETRIES - 1:
                last_err = exc
                time.sleep(_BACKOFF_BASE_S * (2**attempt))
                continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_err = SourceError(f"network error: {exc}", kind="network")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_S * (2**attempt))
                continue
            raise last_err

    raise last_err or SourceError("download failed", kind="network")  # pragma: no cover


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------


def _extract(archive: Path, dest: Path) -> None:
    """Safely extract a ZIP to ``dest`` with zip-bomb + zip-slip guards.

    Raises ``SourceError`` (``not-an-archive`` / ``too-large`` / ``zip-slip``).
    Nothing is written outside ``dest``.
    """
    settings = get_settings()
    max_entries = settings.max_entries
    # The uncompressed cap is read through _int_env (default sourced from
    # Settings) so the per-member streaming counter remains the load-bearing
    # control even when an env override is set mid-process.
    max_uncompressed = _int_env(
        "FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", settings.max_uncompressed_bytes
    )

    if not zipfile.is_zipfile(archive):
        raise SourceError(f"not a ZIP archive: {archive.name}", kind="not-an-archive")

    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise SourceError(
                f"archive has {len(infos)} entries (> {max_entries})", kind="too-large"
            )
        # Cheap fast-reject on the attacker-declared sizes; NOT the authoritative
        # cap (a zip-bomb can lie). The hard cap below counts bytes we actually
        # write while streaming each member.
        declared = sum(info.file_size for info in infos)
        precheck_cap = _int_env(
            "FAUN_SOURCE_MAX_UNCOMPRESSED_BYTES", settings.max_uncompressed_bytes
        )
        if declared > precheck_cap:
            raise SourceError(
                f"archive uncompressed {declared} bytes (> {precheck_cap})",
                kind="too-large",
            )
        # Validate ALL member paths before writing anything (zip-slip).
        for info in infos:
            target = (dest / info.filename).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise SourceError(
                    f"refusing zip member outside dest: {info.filename!r}",
                    kind="zip-slip",
                )
        # Authoritative cap: stream each member and abort once the cumulative
        # bytes actually written exceed the cap.
        written_total = 0
        for info in infos:
            target = (dest / info.filename).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    written_total += len(chunk)
                    if written_total > max_uncompressed:
                        raise SourceError(
                            f"archive uncompressed > {max_uncompressed} bytes",
                            kind="too-large",
                        )
                    out.write(chunk)


def _single_root(directory: Path) -> Path:
    """If ``directory`` contains exactly one sub-directory (and no files), unwrap it.

    A folder ZIP often nests everything under one top-level dir; ingest wants the
    directory that actually holds the WAVs.
    """
    children = list(directory.iterdir())
    dirs = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return directory


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_source(src: str, workdir: Path, *, client: Any = None) -> Path:
    """Resolve ``src`` to a local directory readable by :func:`faun.ingest.scan`.

    Args:
        src: an existing local path, an ``http(s)`` archive URL, or a public
            Yandex.Disk share link (optionally with a ``/A1`` subfolder).
        workdir: job working directory; downloads + extraction live under
            ``workdir/_source/``.
        client: optional ``httpx.Client`` (tests inject a fake). When ``None`` a
            redirect-following client with the configured timeout is created.

    Returns:
        For a local source, ``Path(src)`` UNCHANGED. For a remote source, the
        local extraction directory.

    Raises:
        SourceError: see ``.kind`` taxonomy.
    """
    workdir = Path(workdir)

    # 1) Existing local path -> pass through verbatim (NEVER Path() a URL later).
    local = Path(src)
    if local.exists():
        return local

    if not _is_url(src):
        raise SourceError(
            f"source is neither an existing path nor an http(s) URL: {src!r}",
            kind="not-found",
        )

    # SSRF gate on the *initial* target (Yandex API host or the generic URL).
    initial_target = src
    if _is_yandex_disk(src):
        initial_target = _YADISK_API
    _assert_public_host(initial_target)

    source_root = workdir / "_source"
    archive_dir = source_root / "_dl"
    extract_dir = source_root / "extracted"

    owns_client = client is None
    if owns_client:
        timeout = get_settings().timeout_s
        # follow_redirects=False: we follow + SSRF-validate each hop ourselves so
        # httpx never connects to an internal host on our behalf.
        client = httpx.Client(follow_redirects=False, timeout=timeout)

    try:
        if source_root.exists():
            shutil.rmtree(source_root)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_file = archive_dir / "download.zip"

        is_yandex = _is_yandex_disk(src)
        if is_yandex:
            href = _resolve_yandex_href(src, client)
        else:
            href = src

        _download(href, archive_file, client, is_yandex=is_yandex)
        _extract(archive_file, extract_dir)
        # The unpacked tree is all ingest needs; drop the archive (~tens of GB for
        # real trap folders). Keep the extracted tree — run_pipeline reads it.
        archive_file.unlink(missing_ok=True)
        return _single_root(extract_dir)
    except SourceError:
        if source_root.exists():
            shutil.rmtree(source_root, ignore_errors=True)
        raise
    except Exception as exc:  # normalize anything unexpected
        if source_root.exists():
            shutil.rmtree(source_root, ignore_errors=True)
        raise SourceError(f"failed to resolve source {src!r}: {exc}", kind="network")
    finally:
        if owns_client:
            client.close()


def source_provenance(src: str) -> dict:
    """Return provenance for ``results_meta.json``.

    ``{"source": src, "mode": "local"|"http"|"yadisk"}``. ``mode`` reflects the
    *kind* of source, independent of whether resolution succeeds.
    """
    if _is_yandex_disk(src):
        mode = "yadisk"
    elif _is_url(src):
        mode = "http"
    else:
        mode = "local"
    return {"source": src, "mode": mode}
