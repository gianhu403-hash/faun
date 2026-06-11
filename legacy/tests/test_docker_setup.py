"""Validate Docker setup: per-service Dockerfiles, .dockerignore."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class TestPerServiceDockerfiles:
    @pytest.mark.parametrize(
        "dockerfile",
        ["cloud/Dockerfile", "edge/Dockerfile", "gateway/Dockerfile"],
    )
    def test_dockerfile_exists(self, dockerfile):
        assert (ROOT / dockerfile).is_file(), f"{dockerfile} not found"

    def test_gateway_dockerfile_no_tensorflow(self):
        """Gateway doesn't need TensorFlow — keep it lightweight."""
        content = (ROOT / "gateway" / "Dockerfile").read_text().lower()
        assert "tensorflow" not in content, (
            "Gateway Dockerfile should not install tensorflow"
        )

    def test_gateway_dockerfile_no_texlive(self):
        """Gateway doesn't need texlive — keep it lightweight."""
        content = (ROOT / "gateway" / "Dockerfile").read_text().lower()
        assert "texlive" not in content, "Gateway Dockerfile should not install texlive"


class TestDockerignore:
    def test_dockerignore_exists(self):
        assert (ROOT / ".dockerignore").is_file(), ".dockerignore not found"

    @pytest.mark.parametrize(
        "pattern",
        [".git/", "*.keras", "__pycache__/", "tests/", ".pytest_cache/"],
    )
    def test_dockerignore_excludes(self, pattern):
        content = (ROOT / ".dockerignore").read_text()
        # Check pattern or its prefix is in .dockerignore
        base = pattern.rstrip("/").rstrip("*").rstrip(".")
        assert any(
            base in line for line in content.splitlines() if not line.startswith("#")
        ), f".dockerignore should exclude '{pattern}'"
