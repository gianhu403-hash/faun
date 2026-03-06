import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate
from dataclasses import dataclass
SPEED_OF_SOUND = 343.0  # m/s at ~20°C

@dataclass
class MicPosition:
    lat: float
    lon: float

@dataclass
class TriangulationResult:
    lat: float
    lon: float
    error_m: float

def _latlon_to_meters(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    x = dlon * R * np.cos(np.radians(lat1))
    y = dlat * R
    return x, y

def _meters_to_latlon(lat0, lon0, dx, dy) -> tuple[float, float]:
    R = 6371000
    dlat = np.degrees(dy / R)
    dlon = np.degrees(dx / (R * np.cos(np.radians(lat0))))
    return lat0 + dlat, lon0 + dlon


def _estimate_tdoa(sig_a: np.ndarray, sig_b: np.ndarray, sr: int) -> float:
    corr = correlate(sig_a, sig_b, mode="full")
    lag_samples = corr.argmax() - (len(sig_b) - 1)
    return lag_samples / sr


def triangulate(
    signals: list[np.ndarray],
    mic_positions: list[MicPosition],
    sample_rate: int = 16000,
) -> TriangulationResult:
    assert len(signals) == 3 and len(mic_positions) == 3

    mic_a, mic_b, mic_c = mic_positions
    sig_a, sig_b, sig_c = signals

    lat0, lon0 = mic_a.lat, mic_a.lon

    bx, by = _latlon_to_meters(lat0, lon0, mic_b.lat, mic_b.lon)
    cx, cy = _latlon_to_meters(lat0, lon0, mic_c.lat, mic_c.lon)

    mics_m = np.array([[0, 0], [bx, by], [cx, cy]])

    tdoa_ab = _estimate_tdoa(sig_a, sig_b, sample_rate)
    tdoa_ac = _estimate_tdoa(sig_a, sig_c, sample_rate)

    d_ab = tdoa_ab * SPEED_OF_SOUND  # distance difference A-B
    d_ac = tdoa_ac * SPEED_OF_SOUND  # distance difference A-C

    def cost(pos):
        x, y = pos
        da = np.sqrt(x**2 + y**2)
        db = np.sqrt((x - bx)**2 + (y - by)**2)
        dc = np.sqrt((x - cx)**2 + (y - cy)**2)
        err_ab = (da - db) - d_ab
        err_ac = (da - dc) - d_ac
        return err_ab**2 + err_ac**2

    x0 = mics_m[:, 0].mean()
    y0 = mics_m[:, 1].mean()

    result = minimize(cost, [x0, y0], method="Nelder-Mead",
                      options={"xatol": 0.1, "fatol": 0.01})

    src_x, src_y = result.x
    src_lat, src_lon = _meters_to_latlon(lat0, lon0, src_x, src_y)

    error_m = float(np.sqrt(result.fun)) if result.fun > 0 else 0.0

    return TriangulationResult(lat=src_lat, lon=src_lon, error_m=error_m)
