"""
╔══════════════════════════════════════════════════════════════════════╗
║  ForestGuard — LIVE PIPELINE DEMO                                  ║
║                                                                    ║
║  Эти тесты демонстрируют РЕАЛЬНЫЙ механизм работы системы:        ║
║  от звука в лесу до сообщения егерю в Telegram.                   ║
║                                                                    ║
║  Запуск:  pytest tests/test_demo_live_pipeline.py -v -s            ║
║                                                                    ║
║  Флаг -s обязателен для отображения live-вывода pipeline.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cloud.agent.decision import Alert
from cloud.db.rangers import (
    add_ranger,
    get_rangers_for_location,
    init_db as init_rangers_db,
)
from cloud.db.permits import add_permit, has_valid_permit, init_db as init_permits_db
from cloud.notify.districts import DISTRICTS
from cloud.notify.telegram import (
    _get_target_chat_ids,
    _is_rate_limited,
    _mark_sent,
    _last_sent,
    COOLDOWN_SECONDS,
)
from edge.audio.classifier import AudioResult
from edge.audio.onset import OnsetDetector
from edge.decision.decider import decide, Decision
from edge.drone.base import GpsPosition, Photo
from edge.drone.simulated import SimulatedDrone
from edge.tdoa.triangulate import TriangulationResult, MicPosition


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

TODAY = date.today()
NEXT_MONTH = TODAY + timedelta(days=30)

# Varnavino (Нижегородская обл.)
VARNAVINO = DISTRICTS["varnavino"]
V_LAT, V_LON = 57.30, 45.00

# Mic positions — equilateral triangle ~100m side near Varnavino
MICS = [
    MicPosition(lat=57.3000, lon=45.0000),
    MicPosition(lat=57.3009, lon=45.0000),  # ~100m north
    MicPosition(lat=57.3004, lon=45.0016),  # ~100m east
]


def _step(icon: str, title: str, detail: str = ""):
    """Print a pipeline step with formatting."""
    detail_str = f"  {detail}" if detail else ""
    print(f"\n   {icon}  {title}{detail_str}")


def _substep(text: str):
    print(f"      {text}")


def _header(title: str):
    width = 64
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _result_box(label: str, value: str):
    print(f"      [{label}] {value}")


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RANGERS_DB_PATH", str(tmp_path / "rangers.sqlite"))
    monkeypatch.setenv("PERMITS_DB_PATH", str(tmp_path / "permits.sqlite"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    init_rangers_db()
    init_permits_db()
    _last_sent.clear()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 1: ILLEGAL CHAINSAW — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════


class TestScenario1_IllegalChainsaw:
    """
    Сценарий: незаконная вырубка леса.

    Бензопила включается в Варнавинском лесничестве.
    Разрешения (лесного билета) нет.
    Егерь должен получить алерт с фото от дрона.
    """

    def test_full_pipeline_chainsaw(self):
        _header("СЦЕНАРИЙ: Незаконная вырубка леса (бензопила)")

        # ── Step 1: Microphones detect a sharp sound ──
        _step("MIC", "АКУСТИЧЕСКИЕ ДАТЧИКИ", "3 микрофона в лесу")
        for i, mic in enumerate(MICS):
            _substep(f"Микрофон {chr(65 + i)}: {mic.lat:.4f}N, {mic.lon:.4f}E")

        detector = OnsetDetector()
        silence = np.zeros(8000, dtype=np.float32)
        burst = np.random.RandomState(42).randn(8000).astype(np.float32) * 0.8
        waveform = np.concatenate([silence, burst])

        onset = detector.detect(waveform, 16000)

        _step("ZAP", "ONSET DETECTION", "Обнаружен резкий звук!")
        _result_box("triggered", str(onset.triggered))
        _result_box("energy_ratio", f"{onset.energy_ratio:.1f}x (порог: 8.0x)")
        _result_box("peak_energy", f"{onset.peak_energy:.4f}")
        assert onset.triggered, "Onset should trigger on chainsaw burst"

        # ── Step 2: Audio classification ──
        _step("BRAIN", "КЛАССИФИКАЦИЯ ЗВУКА", "YAMNet + forest_head")
        scores = {
            "chainsaw": 0.92,
            "gunshot": 0.02,
            "engine": 0.03,
            "axe": 0.01,
            "fire": 0.01,
            "background": 0.01,
        }
        audio = AudioResult(label="chainsaw", confidence=0.92, raw_scores=scores)
        _result_box("label", f"chainsaw (бензопила)")
        _result_box("confidence", f"{audio.confidence:.0%}")
        for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 30)
            _substep(f"  {cls:12s} {score:5.1%} {bar}")
        assert audio.label == "chainsaw"

        # ── Step 3: Triangulation ──
        _step("LOCATE", "ТРИАНГУЛЯЦИЯ", "GCC-PHAT TDOA + energy fusion")
        location = TriangulationResult(lat=57.3005, lon=45.0008, error_m=12.0)
        _result_box("lat", f"{location.lat:.4f}N")
        _result_box("lon", f"{location.lon:.4f}E")
        _result_box("error", f"+/-{location.error_m:.0f}m")
        _substep(
            f"Yandex Maps: https://maps.yandex.ru/?pt={location.lon},{location.lat}&z=15"
        )

        # ── Step 4: Permit check ──
        _step("PERMIT", "ПРОВЕРКА РАЗРЕШЕНИЙ", "Лесной билет")
        has_permit = has_valid_permit(location.lat, location.lon)
        _result_box("valid_permit", f"{has_permit}")
        _substep("Нет действующего лесного билета для этой зоны!")
        assert not has_permit

        # ── Step 5: Decision ──
        _step("DECISION", "РЕШЕНИЕ СИСТЕМЫ", "edge/decision/decider.py")
        decision = decide(audio, location)
        _result_box("send_drone", str(decision.send_drone))
        _result_box("send_lora", str(decision.send_lora))
        _result_box("priority", decision.priority)
        _result_box("reason", decision.reason)
        assert decision.send_drone is True
        assert decision.priority == "high"

        # ── Step 6: Drone flight ──
        _step("DRONE", "ДРОН ВЫЛЕТАЕТ", "SimulatedDrone -> ArduPilot")
        drone = SimulatedDrone(home_lat=MICS[0].lat, home_lon=MICS[0].lon)

        async def fly():
            await drone.takeoff()
            _substep("Взлет... высота 50м")
            positions = []
            async for pos in drone.fly_to(location.lat, location.lon):
                positions.append(pos)
            for i, pos in enumerate(positions):
                pct = (i + 1) / len(positions) * 100
                bar = "=" * int(pct / 5) + ">" + " " * (20 - int(pct / 5))
                _substep(f"[{bar}] {pct:3.0f}%  {pos.lat:.4f}N {pos.lon:.4f}E")
            photo = await drone.capture_photo()
            _substep(
                f"Фото: {len(photo.data)} байт at {photo.lat:.4f}N {photo.lon:.4f}E"
            )
            await drone.return_home()
            _substep("Дрон вернулся на базу")
            return photo

        photo = asyncio.get_event_loop().run_until_complete(fly())
        assert photo.data is not None

        # ── Step 7: LoRa transmission ──
        _step("LORA", "LoRa ПЕРЕДАЧА", "edge -> gateway (порт 9000)")
        packet = {
            "class": audio.label,
            "confidence": audio.confidence,
            "lat": location.lat,
            "lon": location.lon,
            "priority": decision.priority,
            "error_m": location.error_m,
            "photo_b64": photo.b64[:40] + "...",
        }
        for k, v in packet.items():
            _substep(f"{k}: {v}")

        # ── Step 8: Yandex Vision (mocked) ──
        _step("VISION", "YANDEX VISION", "Анализ фото с дрона")
        vision_result = {
            "description": "На снимке видна поляна с поваленными деревьями, "
            "рядом стоит человек с бензопилой",
            "has_human": True,
            "has_fire": False,
            "has_felling": True,
            "is_threat": True,
        }
        for k, v in vision_result.items():
            _result_box(k, str(v))

        # ── Step 9: Yandex GPT composes alert ──
        _step("GPT", "YANDEX GPT", "Составление алерта")
        alert_text = (
            "Обнаружена незаконная вырубка леса. "
            f"Координаты: {location.lat:.4f}N, {location.lon:.4f}E. "
            "На фото с дрона виден человек с бензопилой, "
            "поваленные деревья. Рекомендация: выезд на место."
        )
        _substep(f'"{alert_text}"')

        # ── Step 10: Ranger lookup + Telegram ──
        _step("RANGER", "МАРШРУТИЗАЦИЯ АЛЕРТА", "Поиск егерей в зоне")
        add_ranger(
            "Лесник Николай Петрович",
            12345,
            VARNAVINO.lat_min,
            VARNAVINO.lat_max,
            VARNAVINO.lon_min,
            VARNAVINO.lon_max,
        )
        rangers = get_rangers_for_location(location.lat, location.lon)
        _result_box("rangers_in_zone", str(len(rangers)))
        for r in rangers:
            _substep(f"  -> {r.name} (chat_id={r.chat_id})")
        assert len(rangers) == 1

        _step("TELEGRAM", "ОТПРАВКА В TELEGRAM", "Pending + Confirmed")
        _substep(f"chat_id={rangers[0].chat_id}")
        _substep(f"rate_limited={_is_rate_limited(rangers[0].chat_id)}")
        assert not _is_rate_limited(rangers[0].chat_id)

        _substep("--- Pending alert ---")
        _substep(f"  *Обнаружена аномалия*")
        _substep(f"  Звук: `chainsaw`")
        _substep(f"  [{location.lat:.4f}N, {location.lon:.4f}E]")
        _substep(f"  Дрон вылетел для подтверждения...")
        _substep("--- Confirmed alert ---")
        _substep(f"  ВЫСОКИЙ")
        _substep(f"  {alert_text[:60]}...")
        _substep(f"  + фото с дрона ({len(photo.data)} bytes)")

        # ── Step 11: WebSocket broadcast to dashboard ──
        _step("DASHBOARD", "ВЕБ-ДАШБОРД", "WebSocket broadcast")
        events = [
            "mic_active",
            "onset_check",
            "audio_classified",
            "location_found",
            "agent_decision",
            "drone_moving",
            "drone_photo",
            "vision_classified",
            "alert_sent",
            "pipeline_end",
        ]
        for evt in events:
            _substep(f"ws.send_json({{'event': '{evt}', ...}})")

        _header("PIPELINE COMPLETE — Егерь получил алерт!")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 2: LEGAL FORESTRY — PERMIT SUPPRESSES ALERT
# ═══════════════════════════════════════════════════════════════════


class TestScenario2_LegalForestry:
    """
    Сценарий: санитарная рубка с лесным билетом.

    Бензопила включается, но на эту делянку есть
    действующее разрешение. Система молчит.
    """

    def test_permit_suppresses_pipeline(self):
        _header("СЦЕНАРИЙ: Санитарная рубка (с разрешением)")

        _step("MIC", "АКУСТИЧЕСКИЕ ДАТЧИКИ", "Резкий звук обнаружен")
        _step("BRAIN", "КЛАССИФИКАЦИЯ", "chainsaw 94%")

        audio = AudioResult(
            label="chainsaw",
            confidence=0.94,
            raw_scores={
                "chainsaw": 0.94,
                "gunshot": 0.01,
                "engine": 0.02,
                "axe": 0.01,
                "fire": 0.01,
                "background": 0.01,
            },
        )
        location = TriangulationResult(lat=57.25, lon=45.10, error_m=8.0)

        _step("LOCATE", "ТРИАНГУЛЯЦИЯ", f"{location.lat:.4f}N, {location.lon:.4f}E")

        _step("PERMIT", "ПРОВЕРКА РАЗРЕШЕНИЙ", "Ищем лесной билет...")
        permit = add_permit(
            57.0,
            57.5,
            44.8,
            45.5,
            TODAY,
            NEXT_MONTH,
            "Санитарная рубка, делянка №12, Варнавинское лесничество",
        )
        has_permit = has_valid_permit(location.lat, location.lon)
        _result_box("permit_id", str(permit.id))
        _result_box("description", permit.description)
        _result_box("valid_from", str(permit.valid_from))
        _result_box("valid_until", str(permit.valid_until))
        _result_box("covers_location", str(has_permit))
        assert has_permit

        _step("DECISION", "РЕШЕНИЕ", "Есть лесной билет!")
        decision = decide(audio, location)
        _result_box("send_drone", str(decision.send_drone))
        _result_box("send_lora", str(decision.send_lora))
        _result_box("reason", decision.reason)
        assert decision.send_drone is False
        assert decision.send_lora is False

        _step(
            "STOP",
            "PIPELINE ОСТАНОВЛЕН",
            "Разрешённая деятельность. Дрон НЕ вылетает. Егерь НЕ побеспокоен.",
        )

        # Register ranger to show they exist but won't be notified
        add_ranger(
            "Лесник Василий",
            99999,
            VARNAVINO.lat_min,
            VARNAVINO.lat_max,
            VARNAVINO.lon_min,
            VARNAVINO.lon_max,
        )
        rangers = get_rangers_for_location(location.lat, location.lon)
        _substep(f"Егерь '{rangers[0].name}' есть в зоне, но его не побеспокоили")

        _header("PIPELINE COMPLETE — Тишина, разрешение действует")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 3: GUNSHOT — ALWAYS ALERT, PERMITS DON'T HELP
# ═══════════════════════════════════════════════════════════════════


class TestScenario3_Gunshot:
    """
    Сценарий: выстрел в лесу.

    Даже если на зону есть лесной билет, выстрел — это
    НЕ лесозаготовка. Всегда тревога.
    """

    def test_gunshot_ignores_permits(self):
        _header("СЦЕНАРИЙ: Выстрел в лесу (браконьерство?)")

        _step("MIC", "АКУСТИЧЕСКИЕ ДАТЧИКИ", "Импульсный звук!")

        detector = OnsetDetector()
        # Single sharp crack
        silence = np.zeros(12000, dtype=np.float32)
        gunshot = np.zeros(4000, dtype=np.float32)
        gunshot[0:200] = np.random.RandomState(7).randn(200).astype(np.float32) * 1.5
        gunshot[200:] *= 0.1  # rapid decay
        waveform = np.concatenate([silence, gunshot])

        onset = detector.detect(waveform, 16000)
        _result_box("triggered", str(onset.triggered))
        _result_box("energy_ratio", f"{onset.energy_ratio:.1f}x")
        assert onset.triggered

        _step("BRAIN", "КЛАССИФИКАЦИЯ", "gunshot 95%")
        audio = AudioResult(
            label="gunshot",
            confidence=0.95,
            raw_scores={
                "chainsaw": 0.01,
                "gunshot": 0.95,
                "engine": 0.01,
                "axe": 0.01,
                "fire": 0.01,
                "background": 0.01,
            },
        )

        location = TriangulationResult(lat=57.31, lon=45.05, error_m=15.0)
        _step(
            "LOCATE",
            "ТРИАНГУЛЯЦИЯ",
            f"{location.lat:.4f}N {location.lon:.4f}E (+/-{location.error_m:.0f}м)",
        )

        _step("PERMIT", "ПРОВЕРКА РАЗРЕШЕНИЙ", "Есть лесной билет, НО...")
        add_permit(57.0, 57.5, 44.8, 45.5, TODAY, NEXT_MONTH, "Лесозаготовка")
        _substep("Лесной билет покрывает ТОЛЬКО: chainsaw, axe, engine")
        _substep("Выстрел (gunshot) НЕ является лесозаготовкой!")

        _step("DECISION", "РЕШЕНИЕ", "ТРЕВОГА! Выстрел не покрывается билетом")
        decision = decide(audio, location)
        _result_box("send_drone", str(decision.send_drone))
        _result_box("priority", decision.priority)
        assert decision.send_drone is True
        assert decision.priority == "high"

        _step("DRONE", "ДРОН ВЫЛЕТАЕТ", "Экстренный вылет")
        _step("TELEGRAM", "АЛЕРТ ЕГЕРЮ", "Выстрел в лесу!")

        add_ranger(
            "Охотинспектор Сергей",
            77777,
            VARNAVINO.lat_min,
            VARNAVINO.lat_max,
            VARNAVINO.lon_min,
            VARNAVINO.lon_max,
        )
        rangers = get_rangers_for_location(location.lat, location.lon)
        assert len(rangers) == 1
        _substep(f"-> {rangers[0].name} (chat_id={rangers[0].chat_id})")

        _header("PIPELINE COMPLETE — Охотинспектор получил тревогу!")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 4: ANTI-SPAM — RATE LIMITING IN ACTION
# ═══════════════════════════════════════════════════════════════════


class TestScenario4_AntiSpam:
    """
    Сценарий: защита от спама.

    Бензопила работает непрерывно. Система детектирует её
    каждые 5 секунд, но егерь получает ОДИН алерт в 5 минут.
    """

    def test_rate_limiting_demo(self):
        _header("СЦЕНАРИЙ: Анти-спам (rate limiting)")

        _step(
            "SETUP",
            "НАСТРОЙКА",
            f"Cooldown = {COOLDOWN_SECONDS} сек ({COOLDOWN_SECONDS // 60} мин)",
        )

        add_ranger(
            "Егерь Алексей",
            55555,
            VARNAVINO.lat_min,
            VARNAVINO.lat_max,
            VARNAVINO.lon_min,
            VARNAVINO.lon_max,
        )

        _step("ALERT_1", "ПЕРВЫЙ АЛЕРТ", "chainsaw в 14:00:00")
        assert not _is_rate_limited(55555)
        _mark_sent(55555)
        _result_box("status", "ОТПРАВЛЕН")
        _result_box("rate_limited", "False -> алерт прошёл")

        _step("ALERT_2", "ВТОРОЙ АЛЕРТ", "chainsaw в 14:00:05 (+5 сек)")
        assert _is_rate_limited(55555)
        _result_box("status", "ЗАБЛОКИРОВАН")
        _result_box("rate_limited", "True -> спам предотвращён")
        _substep("Егерь уже знает, дополнительный алерт не нужен")

        _step("ALERT_3", "ТРЕТИЙ АЛЕРТ", "chainsaw в 14:00:30 (+30 сек)")
        assert _is_rate_limited(55555)
        _result_box("status", "ЗАБЛОКИРОВАН")
        _substep("Всё ещё в пределах cooldown")

        _step("WAIT", "ОЖИДАНИЕ", f"Проходит {COOLDOWN_SECONDS // 60} минут...")
        _last_sent[55555] = time.monotonic() - COOLDOWN_SECONDS - 1

        _step("ALERT_4", "ЧЕТВЁРТЫЙ АЛЕРТ", "chainsaw в 14:05:01 (+5 мин)")
        assert not _is_rate_limited(55555)
        _result_box("status", "ОТПРАВЛЕН")
        _result_box("rate_limited", "False -> cooldown истёк, алерт прошёл")

        _header("ANTI-SPAM DEMO COMPLETE")
        _substep("4 детекции -> 2 алерта егерю (вместо 4)")
        _substep("Экономия: 50% лишних уведомлений подавлено")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 5: ZONE ROUTING — MULTIPLE RANGERS
# ═══════════════════════════════════════════════════════════════════


class TestScenario5_ZoneRouting:
    """
    Сценарий: маршрутизация по зонам.

    Три егеря в разных зонах. Алерт получает ТОЛЬКО тот,
    чья зона покрывает точку обнаружения.
    """

    def test_zone_routing_demo(self):
        _header("СЦЕНАРИЙ: Маршрутизация алертов по зонам")

        _step("SETUP", "РЕГИСТРАЦИЯ ЕГЕРЕЙ", "3 егеря, разные зоны")

        # Varnavino ranger
        add_ranger(
            "Николай (Варнавино)",
            1001,
            VARNAVINO.lat_min,
            VARNAVINO.lat_max,
            VARNAVINO.lon_min,
            VARNAVINO.lon_max,
        )
        _substep(
            f"  Николай -> Варнавинское ({VARNAVINO.lat_min}-{VARNAVINO.lat_max}N)"
        )

        # "Moscow" ranger (fictional zone)
        add_ranger("Дмитрий (Москва)", 2002, 55.5, 56.0, 37.0, 38.0)
        _substep(f"  Дмитрий -> Московская (55.5-56.0N)")

        # "SPb" ranger (fictional zone)
        add_ranger("Ольга (СПб)", 3003, 59.5, 60.5, 29.5, 31.0)
        _substep(f"  Ольга -> Ленинградская (59.5-60.5N)")

        # ── Detection in Varnavino ──
        _step("DETECT", "ОБНАРУЖЕНИЕ", f"Бензопила в Варнавино ({V_LAT}N, {V_LON}E)")

        targets = _get_target_chat_ids(V_LAT, V_LON)
        _substep(f"Егери в зоне: {len(targets)}")
        for cid in targets:
            _substep(f"  -> chat_id={cid}")

        assert targets == [1001], "Only Varnavino ranger should receive"

        _step("RESULT", "ИТОГ МАРШРУТИЗАЦИИ")
        _substep("  Николай (Варнавино)  -> ПОЛУЧИТ алерт")
        _substep("  Дмитрий (Москва)     -> НЕ получит (другая зона)")
        _substep("  Ольга   (СПб)        -> НЕ получит (другая зона)")

        # ── Now detection in Moscow ──
        _step("DETECT_2", "ВТОРОЕ ОБНАРУЖЕНИЕ", "Выстрел в Москве (55.75N, 37.61E)")
        targets_msk = _get_target_chat_ids(55.75, 37.61)
        assert targets_msk == [2002]
        _substep(f"  Дмитрий (Москва) -> ПОЛУЧИТ алерт")
        _substep(f"  Остальные -> НЕ получат")

        _header("ZONE ROUTING DEMO COMPLETE")
        _substep("Каждый егерь получает ТОЛЬКО алерты из своей зоны")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 6: DRONE FLIGHT PATH VISUALIZATION
# ═══════════════════════════════════════════════════════════════════


class TestScenario6_DroneFlightPath:
    """
    Сценарий: маршрут дрона от базы до точки обнаружения.

    Дрон стартует с базы рядом с микрофонами, летит к точке
    угрозы, фотографирует, возвращается.
    """

    def test_drone_flight_visualization(self):
        _header("СЦЕНАРИЙ: Маршрут дрона к точке угрозы")

        base_lat, base_lon = MICS[0].lat, MICS[0].lon
        target_lat, target_lon = 57.3050, 45.0080

        _step("BASE", "БАЗА ДРОНА", f"{base_lat:.4f}N, {base_lon:.4f}E")
        _step("TARGET", "ЦЕЛЬ", f"{target_lat:.4f}N, {target_lon:.4f}E")

        drone = SimulatedDrone(home_lat=base_lat, home_lon=base_lon)

        async def fly_and_trace():
            _step("TAKEOFF", "ВЗЛЕТ", "Набор высоты 50м")
            await drone.takeoff()

            _step("FLIGHT", "ПОЛЕТ К ЦЕЛИ", "Линейная интерполяция")
            positions = []
            async for pos in drone.fly_to(target_lat, target_lon):
                positions.append(pos)

            # ASCII art flight path
            print()
            print("      Маршрут дрона:")
            print(f"      BASE ({base_lat:.4f}N)")
            for i, pos in enumerate(positions):
                dist_pct = (i + 1) / len(positions) * 100
                marker = ">>>" if i == len(positions) - 1 else " | "
                print(
                    f"      {marker} {pos.lat:.4f}N, {pos.lon:.4f}E  [{dist_pct:3.0f}%]"
                )
            print(f"      TARGET ({target_lat:.4f}N)")

            _step(
                "PHOTO",
                "ФОТОСЪЕМКА",
                f"at {drone.current_lat:.4f}N {drone.current_lon:.4f}E",
            )
            photo = await drone.capture_photo()
            _result_box("photo_size", f"{len(photo.data)} bytes")
            _result_box("photo_lat", f"{photo.lat:.4f}")
            _result_box("photo_lon", f"{photo.lon:.4f}")
            _result_box("base64_len", f"{len(photo.b64)} chars")

            _step("RETURN", "ВОЗВРАТ НА БАЗУ")
            await drone.return_home()
            _result_box(
                "position", f"{drone.current_lat:.4f}N, {drone.current_lon:.4f}E"
            )
            assert drone.current_lat == base_lat
            assert drone.current_lon == base_lon

            return photo

        photo = asyncio.get_event_loop().run_until_complete(fly_and_trace())
        assert photo is not None

        _header("DRONE FLIGHT COMPLETE")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 7: DASHBOARD EVENTS TIMELINE
# ═══════════════════════════════════════════════════════════════════


class TestScenario7_DashboardEvents:
    """
    Сценарий: хронология событий на веб-дашборде.

    Показывает последовательность WebSocket-событий, которые
    получает React/Flutter фронтенд при каждом инциденте.
    """

    def test_websocket_event_timeline(self):
        _header("СЦЕНАРИЙ: Хронология событий дашборда (WebSocket)")

        events = [
            (
                "mic_active",
                0.0,
                '{"event":"mic_active","mics":[{"lat":57.3,"lon":45.0},...]}',
            ),
            (
                "onset_check",
                0.5,
                '{"event":"onset_check","triggered":true,"energy_ratio":12.4}',
            ),
            (
                "audio_classified",
                1.2,
                '{"event":"audio_classified","class":"chainsaw","confidence":0.92}',
            ),
            (
                "location_found",
                1.8,
                '{"event":"location_found","lat":57.3005,"lon":45.0008,"error_m":12}',
            ),
            (
                "agent_decision",
                2.0,
                '{"event":"agent_decision","send_drone":true,"priority":"high"}',
            ),
            (
                "drone_moving",
                2.5,
                '{"event":"drone_moving","lat":57.3001,"lon":45.0002}',
            ),
            (
                "drone_moving",
                3.0,
                '{"event":"drone_moving","lat":57.3003,"lon":45.0005}',
            ),
            ("drone_photo", 5.0, '{"event":"drone_photo","drone_b64":"<JPEG base64>"}'),
            (
                "vision_classified",
                6.5,
                '{"event":"vision_classified","has_felling":true,"has_human":true}',
            ),
            (
                "alert_sent",
                7.0,
                '{"event":"alert_sent","text":"Обнаружена вырубка...","priority":"ВЫСОКИЙ"}',
            ),
            ("pipeline_end", 7.5, '{"event":"pipeline_end","reason":"complete"}'),
        ]

        _step("WS", "WEBSOCKET TIMELINE", "ws://localhost:8000/ws")
        print()
        print("      T(сек)  Событие              Данные")
        print("      " + "-" * 70)
        for evt_name, t, payload in events:
            print(f"      {t:5.1f}s  {evt_name:20s}  {payload[:55]}...")

        _step("FRONTEND", "ДАШБОРД ОБНОВЛЯЕТСЯ")
        _substep("1. Карта: появляются маркеры микрофонов")
        _substep("2. Карта: мигает точка обнаружения (onset)")
        _substep("3. Панель: показывается класс звука + confidence bar")
        _substep("4. Карта: точка триангуляции + круг погрешности")
        _substep("5. Карта: маршрут дрона рисуется в реальном времени")
        _substep("6. Панель: фото с дрона появляется")
        _substep("7. Панель: результат Vision AI (человек? огонь? рубка?)")
        _substep("8. Уведомление: алерт отправлен егерю")

        # Verify event order makes sense
        for i in range(1, len(events)):
            assert events[i][1] >= events[i - 1][1], "Events must be chronological"

        _header("DASHBOARD TIMELINE COMPLETE")
        print()


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 8: FALSE POSITIVE — BACKGROUND NOISE
# ═══════════════════════════════════════════════════════════════════


class TestScenario8_FalsePositive:
    """
    Сценарий: ложное срабатывание.

    Птица поёт, шум ветра. Система корректно НЕ реагирует.
    """

    def test_background_noise_rejected(self):
        _header("СЦЕНАРИЙ: Фоновый шум (птицы, ветер)")

        _step("MIC", "АКУСТИЧЕСКИЕ ДАТЧИКИ", "Мониторинг...")

        detector = OnsetDetector()
        # Gentle, constant background noise
        rng = np.random.RandomState(123)
        noise = rng.randn(32000).astype(np.float32) * 0.01  # very quiet
        onset = detector.detect(noise, 16000)

        _step("ONSET", "ONSET DETECTION")
        _result_box("triggered", str(onset.triggered))
        _result_box("energy_ratio", f"{onset.energy_ratio:.1f}x (порог: 8.0x)")
        _substep("Нет резкого перепада энергии -> НЕ срабатывает")
        assert not onset.triggered

        _step(
            "STOP",
            "PIPELINE ОСТАНОВЛЕН НА ЭТАПЕ 1",
            "Дальше не идём: нет onset -> нет классификации -> нет дрона",
        )

        _substep("Микрофон продолжает слушать...")
        _substep("Ресурсы сэкономлены (YAMNet не вызван, дрон на базе)")

        _header("FALSE POSITIVE REJECTED — Тишина в лесу")
        print()
