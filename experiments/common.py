"""Общие утилиты экспериментов: аудио I/O, окна, метки, метрики, results.csv.

Тяжёлые либы (soundfile, soxr, matplotlib, sklearn) импортируются внутри функций.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

RESULTS_HEADER = [
    "model",
    "dataset",
    "metric",
    "value",
    "runtime_s",
    "vram_mb",
    "notes",
]

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")


# ---------------------------------------------------------------- results.csv


def ensure_results_csv(path: str | Path) -> Path:
    """Создаёт results.csv с заголовком, если файла нет. Возвращает Path."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(RESULTS_HEADER)
    return path


def append_result(
    path: str | Path,
    model: str,
    dataset: str = "",
    metric: str = "",
    value="",
    runtime_s="",
    vram_mb="",
    notes: str = "",
) -> None:
    """Дописывает одну строку результата (создаёт файл с заголовком при нужде)."""
    path = ensure_results_csv(path)
    notes = " | ".join(str(notes).splitlines())[:500]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [model, dataset, metric, value, runtime_s, vram_mb, notes]
        )


# --------------------------------------------------------------------- audio


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True):
    """Читает аудио (soundfile), downmix в mono, ресэмплит через soxr.

    Возвращает (float32 ndarray [n] или [n, ch], sr).
    """
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if mono and x.ndim > 1:
        x = x.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import soxr

        x = soxr.resample(x, sr, target_sr)
        sr = target_sr
    return np.asarray(x, dtype=np.float32), sr


def windows(
    x: np.ndarray, sr: int, win_s: float, hop_s: float | None = None
) -> np.ndarray:
    """Нарезает 1-D сигнал на окна [n_win, win_samples].

    hop_s по умолчанию = win_s (без перекрытия). Неполный хвост отбрасывается;
    если сигнал короче одного окна — дополняется нулями до одного окна.
    """
    x = np.asarray(x)
    win = int(round(win_s * sr))
    hop = int(round((hop_s if hop_s is not None else win_s) * sr))
    if win <= 0 or hop <= 0:
        raise ValueError(f"win/hop must be positive, got win={win} hop={hop}")
    if len(x) < win:
        return np.pad(x, (0, win - len(x)))[None, :]
    n = 1 + (len(x) - win) // hop
    out = np.empty((n, win), dtype=x.dtype)
    for i in range(n):
        out[i] = x[i * hop : i * hop + win]
    return out


def save_spectrogram_png(
    x: np.ndarray, sr: int, out_path: str | Path, n_fft: int = 1024, hop: int = 256
) -> Path:
    """Лог-STFT спектрограмма в PNG (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = windows(x, sr=1, win_s=n_fft, hop_s=hop)  # окна в сэмплах
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)).T
    log_spec = 20.0 * np.log10(spec + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [0, len(x) / sr, 0, sr / 2 / 1000]
    ax.imshow(log_spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("kHz")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------- files / labels


def sample_files(
    directory: str | Path, n: int, seed: int = 42, exts: tuple = AUDIO_EXTS
) -> list[Path]:
    """Сидированная выборка n файлов из каталога (рекурсивно, детерминированно)."""
    files = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in exts)
    if len(files) <= n:
        return files
    return sorted(random.Random(seed).sample(files, n))


def read_bird_labels(csv_path: str | Path) -> dict[str, int]:
    """Метаданные freefield1010 / warblrb10k: itemid -> hasbird (0/1)."""
    labels: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[str(row["itemid"])] = int(row["hasbird"])
    return labels


# ------------------------------------------------------------------- metrics


def auc_score(y_true, y_score) -> float:
    """ROC AUC: sklearn если есть, иначе rank-based (Mann-Whitney) вручную."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        pass

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: only one class present")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(order):  # средние ранги для ничьих
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall(y_true, y_pred) -> tuple[float, float]:
    """(precision, recall) для бинарных меток; 0.0 при пустом знаменателе."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall
