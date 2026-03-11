"""Tests: photo embedding in fpdf2 PDF fallback + photo persistence to DB.

Bug #1: _generate_fpdf2_fallback() ignores drone/ranger photos
Bug #2: handle_inspector_photo() doesn't persist ranger_photo_b64 via update_incident
Bug #3: send_confirmed() doesn't persist drone_photo_b64 via update_incident
"""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cloud.db.incidents import (
    Incident,
    create_incident,
    get_incident,
    update_incident,
    assign_chat_to_incident,
    clear_all_incidents,
)


def _make_test_photo_b64() -> str:
    """Create a minimal valid JPEG as base64 string."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def _clean_incidents():
    clear_all_incidents()
    yield
    clear_all_incidents()


# ---------------------------------------------------------------------------
# Bug #1: fpdf2 fallback must include photos
# ---------------------------------------------------------------------------


class TestFpdf2FallbackPhotos:
    """fpdf2 fallback should embed drone/ranger photos in the PDF."""

    def test_fpdf2_fallback_includes_drone_photo(self):
        from cloud.agent.protocol_pdf import _generate_fpdf2_fallback

        incident = create_incident("chainsaw", 57.3, 45.0, 0.85, "alert")
        incident.drone_photo_b64 = _make_test_photo_b64()

        pdf_bytes = _generate_fpdf2_fallback(incident)

        assert b"/Subtype /Image" in pdf_bytes

    def test_fpdf2_fallback_includes_ranger_photo(self):
        from cloud.agent.protocol_pdf import _generate_fpdf2_fallback

        incident = create_incident("chainsaw", 57.3, 45.0, 0.85, "alert")
        incident.ranger_photo_b64 = _make_test_photo_b64()

        pdf_bytes = _generate_fpdf2_fallback(incident)

        assert b"/Subtype /Image" in pdf_bytes

    def test_fpdf2_fallback_works_without_photos(self):
        from cloud.agent.protocol_pdf import _generate_fpdf2_fallback

        incident = create_incident("chainsaw", 57.3, 45.0, 0.85, "alert")

        pdf_bytes = _generate_fpdf2_fallback(incident)

        assert pdf_bytes[:4] == b"%PDF"
        assert b"/Subtype /Image" not in pdf_bytes


# ---------------------------------------------------------------------------
# Bug #2: ranger_photo_b64 must be persisted via update_incident
# ---------------------------------------------------------------------------


class TestPhotoPersistence:
    """Photos must be persisted to DB via update_incident, not just in memory."""

    @pytest.mark.asyncio
    async def test_ranger_photo_persisted_to_db(self):
        """handle_inspector_photo must call update_incident(ranger_photo_b64=...)."""
        from cloud.notify.bot_handlers import handle_inspector_photo

        incident = create_incident("chainsaw", 57.3, 45.0, 0.85, "alert")
        update_incident(incident.id, status="accepted")
        update_incident(incident.id, status="on_site")

        chat_id = 99999
        assign_chat_to_incident(chat_id, incident.id)

        photo_bytes = b"\xff\xd8fake-jpeg"

        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(photo_bytes))

        mock_photo = MagicMock()
        mock_photo.get_file = AsyncMock(return_value=mock_file)

        mock_update = MagicMock()
        mock_update.effective_chat.id = chat_id
        mock_update.message = AsyncMock()
        mock_update.message.photo = [mock_photo]
        mock_update.message.caption = None

        with patch(
            "cloud.notify.bot_handlers.update_incident",
            wraps=update_incident,
        ) as spy:
            await handle_inspector_photo(mock_update, MagicMock())

        photo_calls = [
            c for c in spy.call_args_list if "ranger_photo_b64" in (c.kwargs or {})
        ]
        assert photo_calls, "update_incident must be called with ranger_photo_b64"

    # -----------------------------------------------------------------------
    # Bug #3: drone_photo_b64 must be persisted via update_incident
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_drone_photo_persisted_to_db(self):
        """send_confirmed must call update_incident(drone_photo_b64=...)."""
        from cloud.agent.decision import Alert
        from cloud.notify.telegram import send_confirmed

        # Stored incident (the DB reference)
        stored = create_incident("chainsaw", 57.3, 45.0, 0.85, "alert")

        # Separate object with same id — NOT the stored reference
        separate = Incident(
            id=stored.id,
            audio_class="chainsaw",
            lat=57.3,
            lon=45.0,
            confidence=0.85,
            gating_level="alert",
        )

        alert = Alert(
            text="Detected chainsaw",
            priority="HIGH",
            lat=57.3,
            lon=45.0,
        )
        photo_bytes = b"\xff\xd8fake-jpeg"

        await send_confirmed(alert, photo_bytes, separate)

        # The STORED incident must have the photo (only possible via update_incident)
        result = get_incident(stored.id)
        assert result.drone_photo_b64 is not None, (
            "drone_photo_b64 must be persisted via update_incident"
        )
