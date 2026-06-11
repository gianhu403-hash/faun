"""Generate sample incident data for DataLens dashboard.

Creates a CSV with 200 realistic forest monitoring incidents
in the Varnavino forestry district (Nizhny Novgorod Oblast).

Response-time modelling accounts for:
  - Violation urgency (fire/gunshot → faster dispatch)
  - Time of day (night → slower)
  - Distance to nearest ranger post (further → slower)
  - Random variation

Usage:
    python -m cloud.analytics.sample_incidents
"""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

CLASSES = ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]
CLASS_WEIGHTS = [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]

STATUSES = ["resolved", "false_alarm", "pending"]
STATUS_WEIGHTS = [0.60, 0.25, 0.15]

GATING_LEVELS = ["alert", "verify", "log"]

DISTRICTS = [
    "Мдальское",
    "Семёнборское",
    "Поплывинское",
    "Каменниковское",
    "Варнавинское",
    "Колесниковское",
    "Камешное",
    "Кайское",
]

# Approximate ranger post positions (lat, lon) within each sub-district
RANGER_POSTS: dict[str, tuple[float, float]] = {
    "Мдальское": (57.475, 44.70),
    "Семёнборское": (57.425, 44.90),
    "Поплывинское": (57.375, 45.10),
    "Каменниковское": (57.275, 44.70),
    "Варнавинское": (57.225, 44.90),
    "Колесниковское": (57.175, 45.10),
    "Камешное": (57.125, 45.20),
    "Кайское": (57.125, 45.30),
}

RANGER_NAMES = [
    "Козлов А.С.",
    "Петров И.В.",
    "Сидорова Е.М.",
    "Васильев Д.А.",
    "Кузнецов П.Н.",
    "Морозов Г.Л.",
    "Новикова О.Р.",
    "Соколов В.Ю.",
]

# Varnavino forestry zone bounding box
LAT_MIN, LAT_MAX = 57.0, 57.5
LON_MIN, LON_MAX = 44.5, 45.5

OUTPUT_PATH = Path(__file__).parent / "sample_incidents.csv"
NUM_INCIDENTS = 200


def _random_timestamp(rng: random.Random, days_back: int = 30) -> datetime:
    """Return a random datetime within *days_back* days before 2026-03-08."""
    base = datetime(2026, 3, 8, 12, 0, 0)
    offset = timedelta(
        days=rng.randint(0, days_back),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )
    return base - offset


def _confidence_for_class(rng: random.Random, cls: str) -> float:
    if cls == "background":
        return round(rng.uniform(0.30, 0.55), 3)
    return round(rng.uniform(0.60, 0.98), 3)


def _gating_for_confidence(conf: float) -> str:
    if conf >= 0.85:
        return "alert"
    if conf >= 0.65:
        return "verify"
    return "log"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Base response time range (minutes) by violation class.
# High-urgency events (fire, gunshot) get faster dispatch.
_BASE_RESPONSE: dict[str, tuple[float, float]] = {
    "fire": (5, 15),
    "gunshot": (6, 18),
    "chainsaw": (10, 30),
    "axe": (12, 35),
    "engine": (8, 25),
    "background": (5, 15),
}


def _response_time_min(
    rng: random.Random,
    cls: str,
    lat: float,
    lon: float,
    district: str,
    ts: datetime,
) -> float:
    """Generate a realistic response time in minutes.

    Components:
      1. Base interval from violation urgency
      2. Distance penalty: ~2 min per km from ranger post
      3. Night penalty (22:00–06:00): +30–60 %
      4. Gaussian noise (±20 %)
    """
    lo, hi = _BASE_RESPONSE.get(cls, (15, 50))
    base = rng.uniform(lo, hi)

    # Distance from nearest ranger post
    post = RANGER_POSTS.get(district)
    if post:
        dist_km = _haversine_km(lat, lon, post[0], post[1])
    else:
        dist_km = rng.uniform(2, 8)
    travel = dist_km * rng.uniform(0.5, 1.0)  # ~0.7 min/km with vehicles

    # Night-time penalty
    hour = ts.hour
    night_factor = 1.0
    if hour >= 22 or hour < 6:
        night_factor = rng.uniform(1.15, 1.35)

    raw = (base + travel) * night_factor
    # Add noise ±20 %
    noisy = raw * rng.uniform(0.8, 1.2)
    return round(max(5, noisy), 1)


def generate_incidents(n: int = NUM_INCIDENTS, seed: int = 42) -> list[dict]:
    """Generate *n* sample incidents with realistic response times."""
    rng = random.Random(seed)
    rows: list[dict] = []

    for i in range(1, n + 1):
        cls = rng.choices(CLASSES, weights=CLASS_WEIGHTS, k=1)[0]
        conf = _confidence_for_class(rng, cls)
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        if cls == "background":
            status = "false_alarm"

        lat = round(rng.uniform(LAT_MIN, LAT_MAX), 6)
        lon = round(rng.uniform(LON_MIN, LON_MAX), 6)
        district = rng.choice(DISTRICTS)
        ts = _random_timestamp(rng)

        if status in ("resolved", "false_alarm"):
            response_time = _response_time_min(rng, cls, lat, lon, district, ts)
            ranger_name = rng.choice(RANGER_NAMES)
            resolution = (
                "Протокол составлен, материалы переданы"
                if status == "resolved"
                else "Ложное срабатывание, закрыто"
            )
        else:
            response_time = None
            ranger_name = None
            resolution = "Ожидает реакции"

        rows.append(
            {
                "id": i,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "lat": lat,
                "lon": lon,
                "audio_class": cls,
                "confidence": conf,
                "gating_level": _gating_for_confidence(conf),
                "status": status,
                "district": district,
                "response_time_min": response_time,
                "ranger_name": ranger_name,
                "resolution_details": resolution,
            }
        )
    return rows


FIELDNAMES = [
    "id",
    "timestamp",
    "lat",
    "lon",
    "audio_class",
    "confidence",
    "gating_level",
    "status",
    "district",
    "response_time_min",
    "ranger_name",
    "resolution_details",
]


def write_csv(rows: list[dict], path: Path = OUTPUT_PATH) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def get_incidents_csv_text() -> str:
    """Return CSV content as string (for API export)."""
    rows = generate_incidents()
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    rows = generate_incidents()
    write_csv(rows)
    print(f"Generated {len(rows)} incidents -> {OUTPUT_PATH}")

    # Print summary stats
    resolved = [r for r in rows if r["response_time_min"] is not None]
    times = [r["response_time_min"] for r in resolved]
    print(f"  Responded: {len(resolved)}/{len(rows)}")
    print(f"  Avg response time: {sum(times)/len(times):.1f} min")
    print(f"  Min: {min(times):.1f} min, Max: {max(times):.1f} min")

    by_class: dict[str, list[float]] = {}
    for r in resolved:
        by_class.setdefault(r["audio_class"], []).append(r["response_time_min"])
    print("  By class:")
    for cls, t in sorted(by_class.items(), key=lambda x: sum(x[1]) / len(x[1])):
        print(f"    {cls:12s}: avg {sum(t)/len(t):.1f} min ({len(t)} incidents)")
