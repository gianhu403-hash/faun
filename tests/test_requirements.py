"""Validate requirements.txt: upper bounds on critical deps, no stdlib packages."""

from pathlib import Path

import pytest

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"

CRITICAL_DEPS = ["tensorflow", "numpy", "python-telegram-bot", "scipy"]
STDLIB_PACKAGES = ["asyncio", "os", "sys", "json", "logging"]


def _parse_requirements():
    """Return list of non-empty, non-comment lines from requirements.txt."""
    return [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class TestRequirementsBounds:
    @pytest.mark.parametrize("dep", CRITICAL_DEPS)
    def test_critical_dep_has_upper_bound(self, dep):
        """Critical dependencies must have an upper bound (<) to prevent breaking upgrades."""
        lines = _parse_requirements()
        matching = [
            l
            for l in lines
            if l.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip() == dep
        ]
        assert matching, f"{dep} not found in requirements.txt"
        line = matching[0]
        assert "<" in line, (
            f"{dep} has no upper bound: '{line}'. "
            f"Add e.g. ',<X.Y' to prevent breaking upgrades."
        )


class TestNoStdlib:
    def test_no_stdlib_in_requirements(self):
        """Stdlib modules must not appear in requirements.txt."""
        lines = _parse_requirements()
        for pkg in STDLIB_PACKAGES:
            stdlib_lines = [l for l in lines if l.strip() == pkg]
            assert not stdlib_lines, (
                f"Stdlib package '{pkg}' found in requirements.txt — remove it"
            )
