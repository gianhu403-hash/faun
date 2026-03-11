"""TDD tests for LaTeX code path in protocol_pdf.

Behaviors:
1. generate_protocol dispatches to LaTeX when lualatex is on PATH
2. Jinja2 template renders act_patrol.tex with custom delimiters
3. _build_context produces all required template fields
4. _compile_latex raises RuntimeError on lualatex failure
5. generate_protocol falls back to fpdf2 when lualatex missing
"""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from cloud.agent.protocol_pdf import (
    _build_context,
    _compile_latex,
    _make_jinja_env,
    generate_protocol,
)
from cloud.db.incidents import Incident

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_PDF = b"%PDF-1.5 fake"


def _make_incident(**overrides) -> Incident:
    """Create a minimal Incident for testing."""
    defaults = dict(
        id="abcd1234-0000-0000-0000-000000000000",
        audio_class="chainsaw",
        lat=57.3456,
        lon=45.1234,
        confidence=0.92,
        gating_level="alert",
        status="pending",
        created_at=time.mktime((2026, 3, 10, 14, 30, 0, 0, 0, -1)),
        accepted_by_name="Иванов И.И.",
        accepted_by_chat_id=None,
        drone_photo_b64=None,
        drone_comment="Видны следы рубки",
        ranger_photo_b64=None,
        ranger_report_raw=None,
        ranger_report_legal="Обнаружена незаконная рубка",
    )
    defaults.update(overrides)
    return Incident(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLatexPath:
    def test_generate_protocol_uses_latex_when_lualatex_available(self):
        """When lualatex is on PATH, generate_protocol takes the LaTeX branch."""
        inc = _make_incident()

        with (
            patch(
                "cloud.agent.protocol_pdf.shutil.which",
                return_value="/usr/bin/lualatex",
            ),
            patch(
                "cloud.agent.protocol_pdf._compile_latex", return_value=FAKE_PDF
            ) as mock_compile,
            patch("cloud.agent.protocol_pdf.get_ranger_by_chat_id", return_value=None),
        ):
            result = generate_protocol(inc, "ст. 260 УК РФ")

        assert result == FAKE_PDF
        mock_compile.assert_called_once()

    def test_generate_protocol_falls_back_to_fpdf2(self):
        """When lualatex is missing, generate_protocol uses fpdf2 fallback."""
        inc = _make_incident()

        with (
            patch("cloud.agent.protocol_pdf.shutil.which", return_value=None),
            patch(
                "cloud.agent.protocol_pdf._generate_fpdf2_fallback",
                return_value=FAKE_PDF,
            ) as mock_fpdf2,
        ):
            result = generate_protocol(inc)

        assert result == FAKE_PDF
        mock_fpdf2.assert_called_once_with(inc, "")


class TestJinja2Template:
    def test_jinja2_template_renders_act_patrol_tex(self):
        """act_patrol.tex renders with custom delimiters and sample context."""
        from cloud.agent.protocol_pdf import _TEMPLATE_DIR

        env = _make_jinja_env(str(_TEMPLATE_DIR))
        template = env.get_template("act_patrol.tex")

        context = _build_context(
            _make_incident(),
            legal_articles="ст. 260 УК РФ",
        )
        # Provide image paths as empty (no images)
        context["drone_photo_path"] = ""
        context["ranger_photo_path"] = ""
        context["font_path"] = ""

        rendered = template.render(**context)

        # Key strings must appear in rendered output
        assert "АКТ" in rendered
        assert "патрулирования лесов" in rendered
        assert "ABCD1234" in rendered  # act_number
        assert "57.3456" in rendered  # lat
        assert "45.1234" in rendered  # lon
        assert r"ст. 260 УК РФ" in rendered
        assert "Иванов И.И." in rendered
        assert r"\documentclass" in rendered


class TestBuildContext:
    def test_build_context_contains_all_template_fields(self):
        """_build_context returns all keys needed by act_patrol.tex."""
        inc = _make_incident()

        with patch("cloud.agent.protocol_pdf.get_ranger_by_chat_id", return_value=None):
            ctx = _build_context(inc, "ст. 260 УК РФ")

        expected_keys = {
            "act_number",
            "act_day",
            "act_month",
            "act_year",
            "patrol_date",
            "patrol_time_start",
            "patrol_time_end",
            "lat",
            "lon",
            "sub_district",
            "quarter",
            "compartment",
            "forest_purpose",
            "ranger_name",
            "badge_number",
            "violation_type",
            "confidence",
            "gating_level",
            "detected_at",
            "incident_id",
            "article",
            "legal_articles",
            "drone_comment",
            "ranger_report",
            "drone_photo_path",
            "ranger_photo_path",
            "font_path",
        }
        assert expected_keys.issubset(set(ctx.keys())), (
            f"Missing keys: {expected_keys - set(ctx.keys())}"
        )

        # Verify formatting
        assert ctx["act_number"] == "ABCD1234"
        assert ctx["act_month"] == "марта"
        assert ctx["confidence"] == "92"
        assert ctx["lat"] == "57.3456"
        assert ctx["lon"] == "45.1234"
        assert ctx["ranger_name"] == "Иванов И.И."
        assert ctx["legal_articles"] == "ст. 260 УК РФ"


class TestCompileLatex:
    def test_compile_latex_raises_on_failure(self, tmp_path):
        """_compile_latex raises RuntimeError when lualatex returns non-zero."""
        tex_file = tmp_path / "test.tex"
        tex_file.write_text(
            r"\documentclass{article}\begin{document}Hello\end{document}"
        )

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = "! Emergency stop."

        with patch("cloud.agent.protocol_pdf.subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="lualatex failed"):
                _compile_latex(str(tex_file), str(tmp_path), runs=1)
