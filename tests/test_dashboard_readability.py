"""Tests: Dashboard readability — regression guard for CSS sizing.

Verifies minimum thresholds for font sizes, panel widths, and element
dimensions to prevent accidental regressions to hard-to-read values.
"""

import re
from pathlib import Path

import pytest

INDEX_HTML = (
    Path(__file__).resolve().parent.parent / "cloud" / "interface" / "index.html"
)


@pytest.fixture
def html():
    return INDEX_HTML.read_text()


def _css_px(html: str, pattern: str) -> int:
    """Extract a pixel value from CSS using a regex pattern."""
    m = re.search(pattern, html)
    assert m, f"Pattern not found: {pattern}"
    return int(m.group(1))


def test_sidebar_width(html):
    """Sidebar must be at least 460px wide for readability."""
    width = _css_px(html, r"grid-template-columns:\s*1fr\s+(\d+)px")
    assert width >= 460, f"Sidebar width {width}px is too narrow (min 460px)"


def test_events_log_size(html):
    """Events log must be wide and tall enough to read comfortably."""
    width = _css_px(html, r"\.events-log\s*\{[^}]*width:\s*(\d+)px")
    max_h = _css_px(html, r"\.events-log\s*\{[^}]*max-height:\s*(\d+)px")
    assert width >= 520, f"Events-log width {width}px too narrow (min 520px)"
    assert max_h >= 260, f"Events-log max-height {max_h}px too short (min 260px)"


def test_events_log_font(html):
    """Events log text must be at least 14px for readability."""
    font = _css_px(html, r"\.ev-line\s*\{[^}]*font-size:\s*(\d+)px")
    assert font >= 14, f"Events-log font {font}px too small (min 14px)"


def test_sidebar_fonts(html):
    """Key sidebar fonts must meet minimum readability thresholds."""
    detect = _css_px(html, r"\.detect-class\s*\{[^}]*font-size:\s*(\d+)px")
    label = _css_px(html, r"\.sb-label\s*\{[^}]*font-size:\s*(\d+)px")
    gpt = _css_px(html, r"\.gpt-text\s*\{[^}]*font-size:\s*(\d+)px")
    assert detect >= 20, f"detect-class font {detect}px too small (min 20px)"
    assert label >= 12, f"sb-label font {label}px too small (min 12px)"
    assert gpt >= 15, f"gpt-text font {gpt}px too small (min 15px)"


def test_header_size(html):
    """Header must be tall enough and logo readable."""
    height = _css_px(html, r"\.header\s*\{[^}]*height:\s*(\d+)px")
    logo = _css_px(html, r"\.logo\s*\{[^}]*font-size:\s*(\d+)px")
    assert height >= 48, f"Header height {height}px too short (min 48px)"
    assert logo >= 18, f"Logo font {logo}px too small (min 18px)"
