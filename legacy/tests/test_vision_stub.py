"""RED-phase tests for the vision stub fallback.

`_stub_result()` is invoked when no real vision API is configured (or all
upstream calls fail). It must NOT report a confirmed felling/threat — that
would cause the decision engine to escalate based on no evidence at all.
"""

from cloud.vision.classifier import _stub_result


def test_stub_result_is_not_threat() -> None:
    """Stub must not claim felling/threat — there is no evidence."""
    result = _stub_result()
    assert result.is_threat is False
    assert result.has_felling is False


def test_stub_result_text_indicates_unavailable() -> None:
    """Operators must see clearly that vision is unavailable."""
    result = _stub_result()
    assert "недоступ" in result.description.lower()
