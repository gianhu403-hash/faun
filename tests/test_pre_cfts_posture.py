"""Pre-CFTS posture gate (FAUN-33).

Encodes the audit checklist as reproducible assertions:
- README.md is in Russian (target audience)
- No hardcoded secrets in tracked files
- .env.example contains placeholders only (no real values leaking via examples)
- .gitignore covers .env* family (so .env.local etc. can't slip in)

Demo-blocker checks (vision stub safety, etc.) live with the relevant code's
own test files.

Re-run before any public showing: `pytest tests/test_pre_cfts_posture.py -v`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_in_russian():
    """README.md must be in Russian — primary audience for this project."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", text))
    ratio = cyrillic / max(len(text), 1)
    assert ratio > 0.3, (
        f"README.md cyrillic ratio {ratio:.2%} — must be >30% (Russian content)"
    )


def test_no_hardcoded_secrets_in_tracked_files():
    """Tracked files must not contain hardcoded API keys / tokens / passwords.

    Catches: FOO_KEY = "actual-value-here". Allows: os.getenv("FOO_KEY").
    """
    out = (
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
        .strip()
        .splitlines()
    )
    pattern = re.compile(
        r"(api[_-]?key|token|secret|password)\s*=\s*['\"]([^'\"<\n]{8,})", re.I
    )
    suspicious_extensions = (".py", ".yml", ".yaml", ".env", ".sh")
    findings: list[str] = []
    for f in out:
        if not f.endswith(suspicious_extensions):
            continue
        try:
            content = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, IsADirectoryError):
            continue
        for m in pattern.finditer(content):
            value = m.group(2)
            # Allow placeholders, env-var references, and obvious test fixtures
            if value in (
                "your_key_here",
                "<placeholder>",
                "xxx",
                "test",
                "test-token",
                "fake-token",
            ):
                continue
            if "getenv" in m.group(0) or "environ" in m.group(0):
                continue
            findings.append(f"{f}: {m.group(0)[:60]}...")
    assert not findings, "Possible hardcoded secrets:\n" + "\n".join(findings)


def test_dotenv_example_has_placeholder_only():
    """Each non-comment KEY=VALUE in .env.example must have empty/placeholder VALUE."""
    text = (ROOT / ".env.example").read_text()
    bad: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if value == "" or value.startswith("<") or value == "0":
            continue
        if value.startswith("your_") or value in ("placeholder", "xxx"):
            continue
        bad.append(f"{key.strip()}={value}")
    assert not bad, ".env.example must have empty placeholders only:\n" + "\n".join(bad)


def test_dotenv_pattern_in_gitignore():
    """`.env*` (not just `.env`) must be in .gitignore so .env.local etc. are caught."""
    text = (ROOT / ".gitignore").read_text()
    has_pattern = any(
        line.strip() in (".env", ".env*", ".env.*", "*.env")
        for line in text.splitlines()
    )
    assert has_pattern, ".gitignore must include .env or .env* pattern"
