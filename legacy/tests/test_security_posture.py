"""Security posture tests for FAUN-37.

Structural assertions that encode security hardening requirements as tests.
Each test maps to one Acceptance Criterion. Failures here mean regression
in security posture, not in business logic.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_cloud_dockerfile_has_user_faun():
    text = (ROOT / "cloud" / "Dockerfile").read_text()
    assert re.search(r"^USER faun\b", text, re.MULTILINE), (
        "cloud/Dockerfile must declare USER faun before CMD"
    )


def test_edge_dockerfile_has_user_faun():
    text = (ROOT / "edge" / "Dockerfile").read_text()
    assert re.search(r"^USER faun\b", text, re.MULTILINE), (
        "edge/Dockerfile must declare USER faun before CMD"
    )


def test_gateway_dockerfile_has_user_faun():
    text = (ROOT / "gateway" / "Dockerfile").read_text()
    assert re.search(r"^USER faun\b", text, re.MULTILINE), (
        "gateway/Dockerfile must declare USER faun before CMD"
    )


def test_ci_yml_runs_bandit_and_pip_audit():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "bandit" in text, "ci.yml must run bandit (SAST)"
    assert "pip-audit" in text, "ci.yml must run pip-audit (SCA)"


def test_deploy_yml_pins_ssh_action_to_sha():
    text = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert "appleboy/ssh-action@v1" not in text, (
        "deploy.yml must not use floating tag @v1 — pin to commit SHA"
    )
    assert re.search(r"appleboy/ssh-action@[a-f0-9]{40}", text), (
        "deploy.yml must pin appleboy/ssh-action to a 40-char commit SHA"
    )


def test_exception_handler_does_not_leak_str_exc():
    text = (ROOT / "cloud" / "interface" / "main.py").read_text()
    # Look for the exception handler block
    assert re.search(r"@app\.exception_handler\(Exception\)", text), (
        "exception handler must exist"
    )
    # The body must NOT contain str(exc) being returned
    assert '"error": str(exc)' not in text and "'error': str(exc)" not in text, (
        "exception handler must not return str(exc) — leaks internals"
    )
    # And should contain a generic message
    assert '"error": "internal"' in text or "'error': 'internal'" in text, (
        "exception handler should return generic 'internal' error"
    )


def test_ydb_microphones_uses_declare_not_fstring_limit():
    text = (ROOT / "cloud" / "db" / "ydb_microphones.py").read_text()
    # No f-string LIMIT
    assert not re.search(r"""f["'][^"']*LIMIT\s*\{""", text), (
        "ydb_microphones.py must not use f-string LIMIT — use DECLARE $limit"
    )
    # Should use DECLARE (parameterized)
    assert "DECLARE $limit" in text, (
        "ydb_microphones.py should use DECLARE $limit AS Uint64 pattern"
    )


def test_env_example_documents_security_envs():
    text = (ROOT / ".env.example").read_text()
    for required in [
        "FAUN_API_KEY",
        "ALLOWED_RANGER_CHAT_IDS",
        "ALLOWED_DRONE_CHAT_IDS",
    ]:
        assert required in text, f".env.example must mention {required}"
