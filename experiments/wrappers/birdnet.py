"""BirdNET через birdnetlib (CPU, образ faun-ml-cpu).

analyze_file / analyze_array -> список детекций (start, end, species, conf).
"""

from __future__ import annotations

from pathlib import Path

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from birdnetlib.analyzer import Analyzer

        _analyzer = Analyzer()
    return _analyzer


def analyze_file(
    path: str | Path,
    min_conf: float = 0.1,
    lat: float | None = None,
    lon: float | None = None,
    date=None,
) -> list[tuple[float, float, str, float]]:
    """Анализ WAV-файла. Возвращает [(start_s, end_s, common_name, confidence)]."""
    from birdnetlib import Recording

    kwargs = {"min_conf": min_conf}
    if lat is not None and lon is not None:
        kwargs.update(lat=lat, lon=lon)
    if date is not None:
        kwargs["date"] = date

    rec = Recording(_get_analyzer(), str(path), **kwargs)
    rec.analyze()
    return [
        (
            float(d["start_time"]),
            float(d["end_time"]),
            d.get("common_name", d.get("scientific_name", "?")),
            float(d["confidence"]),
        )
        for d in rec.detections
    ]


def analyze_array(
    x, sr: int, min_conf: float = 0.1, **kwargs
) -> list[tuple[float, float, str, float]]:
    """Анализ массива: пишет временный WAV и зовёт analyze_file."""
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, x, sr)
        return analyze_file(tmp.name, min_conf=min_conf, **kwargs)
