"""Tests for RAG agent — SDK timeout, fallback, and class-aware endpoint fallback."""

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from cloud.agent.rag_agent import (
    _call_yandex_with_sdk,
    query_rag_enriched,
    IncidentContext,
)


# ---------------------------------------------------------------------------
# Behavior 1: SDK timeout → fallback to plain API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_timeout_falls_back_to_plain_api():
    """When SDK sync call exceeds SDK_TIMEOUT, _call_yandex_with_sdk
    should catch TimeoutError and fall back to _call_yandex_plain."""

    def slow_sdk_sync(prompt):
        time.sleep(2)
        return "sdk answer (too late)"

    with (
        patch(
            "cloud.agent.rag_agent._call_yandex_with_sdk_sync",
            side_effect=slow_sdk_sync,
        ),
        patch("cloud.agent.rag_agent.SDK_TIMEOUT", 1),
        patch(
            "cloud.agent.rag_agent._call_yandex_plain",
            new_callable=AsyncMock,
            return_value="plain answer",
        ) as mock_plain,
    ):
        result = await _call_yandex_with_sdk("test prompt")

    assert result == "plain answer"
    mock_plain.assert_awaited_once()


# ---------------------------------------------------------------------------
# Behavior 2: SDK success → returns SDK answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_success_returns_sdk_answer():
    """When SDK returns quickly, its answer is used and plain API is NOT called."""

    with (
        patch(
            "cloud.agent.rag_agent._call_yandex_with_sdk_sync",
            return_value="sdk answer",
        ),
        patch("cloud.agent.rag_agent.SDK_TIMEOUT", 5),
        patch(
            "cloud.agent.rag_agent._call_yandex_plain",
            new_callable=AsyncMock,
            return_value="plain answer",
        ) as mock_plain,
    ):
        result = await _call_yandex_with_sdk("test prompt")

    assert result == "sdk answer"
    mock_plain.assert_not_awaited()


# ---------------------------------------------------------------------------
# Behavior 3: SDK exception → fallback to plain API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_exception_falls_back_to_plain_api():
    """When SDK raises an exception, fall back to plain API."""

    with (
        patch(
            "cloud.agent.rag_agent._call_yandex_with_sdk_sync",
            side_effect=RuntimeError("SDK broke"),
        ),
        patch("cloud.agent.rag_agent.SDK_TIMEOUT", 5),
        patch(
            "cloud.agent.rag_agent._call_yandex_plain",
            new_callable=AsyncMock,
            return_value="plain fallback",
        ) as mock_plain,
    ):
        result = await _call_yandex_with_sdk("test prompt")

    assert result == "plain fallback"
    mock_plain.assert_awaited_once()


# ---------------------------------------------------------------------------
# Behavior 4: E2E — query_rag_enriched uses plain on SDK timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_rag_enriched_uses_plain_on_sdk_timeout():
    """End-to-end: query_rag_enriched returns plain answer when SDK times out,
    and completes in reasonable time (< 3s)."""

    def slow_sdk_sync(prompt):
        time.sleep(2)
        return "too late"

    mock_forest = MagicMock(
        quarter_number="42",
        sub_district="Варнавинское",
        species_composition="сосна 80%, ель 20%",
        zone_type="эксплуатационная",
        area_ha=15.0,
    )

    with (
        patch(
            "cloud.agent.rag_agent._call_yandex_with_sdk_sync",
            side_effect=slow_sdk_sync,
        ),
        patch("cloud.agent.rag_agent.SDK_TIMEOUT", 1),
        patch("cloud.agent.rag_agent.SEARCH_INDEX_ID", "test-index"),
        patch(
            "cloud.agent.rag_agent._call_yandex_plain",
            new_callable=AsyncMock,
            return_value="enriched plain answer",
        ),
        patch(
            "cloud.integrations.fgis_lk.fgis_client.get_forest_unit",
            return_value=mock_forest,
        ),
        patch("cloud.db.permits.has_valid_permit", return_value=False),
    ):
        ctx = IncidentContext(
            audio_class="chainsaw",
            confidence=0.9,
            lat=57.3,
            lon=44.6,
        )
        start = time.monotonic()
        result = await query_rag_enriched(ctx)
        elapsed = time.monotonic() - start

    assert result == "enriched plain answer"
    assert elapsed < 3, f"query_rag_enriched took {elapsed:.1f}s, expected < 3s"


# ---------------------------------------------------------------------------
# Behavior 5: Endpoint timeout fallback — class-aware articles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_timeout_fallback_is_class_aware():
    """When the endpoint catches TimeoutError, the static fallback should
    mention articles relevant to the audio_class, not generic ones."""
    from httpx import ASGITransport, AsyncClient
    from cloud.interface.main import app

    with patch(
        "cloud.interface.main.query_rag_enriched",
        new_callable=AsyncMock,
        side_effect=asyncio.TimeoutError,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/rag-query",
                json={
                    "question": "Что делать?",
                    "audio_class": "gunshot",
                    "confidence": 0.9,
                    "lat": 57.3,
                    "lon": 44.6,
                },
            )

    assert resp.status_code == 200
    body = resp.json()["answer"]
    # gunshot → ст. 258 УК РФ (браконьерство), NOT ст. 260 (рубка)
    assert "258" in body, f"Expected ст. 258 for gunshot, got: {body[:200]}"
    assert "260" not in body, (
        f"Should not contain ст. 260 for gunshot, got: {body[:200]}"
    )
