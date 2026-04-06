#!/usr/bin/env python3
"""Download diverse demo sounds for live classification demo.

Downloads 3 different real audio samples per class (5 classes × 3 = 15 files)
into demo/sounds/{scenario}/{1,2,3}.wav.

Sources:
  - ESC-50 (GitHub): chainsaw, engine, fire, axe (door_wood_knock)
  - UrbanSound8K (HuggingFace parquet): gunshot

Output: WAV, 16kHz, mono, 5s, peak-normalized (0.95).

Usage:
    python demo/download_sounds.py          # download (skip existing)
    python demo/download_sounds.py --force  # re-download all
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

SOUNDS_DIR = Path(__file__).parent / "sounds"
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION
N_PER_CLASS = 3

ESC50_META_URL = "https://github.com/karolpiczak/ESC-50/raw/master/meta/esc50.csv"
ESC50_AUDIO_BASE = "https://github.com/karolpiczak/ESC-50/raw/master/audio"

# ESC-50 target class IDs
ESC50_TARGETS: dict[str, int] = {
    "chainsaw": 41,
    "engine": 44,
    "fire": 12,  # crackling_fire
    "axe": 30,  # door_wood_knock
}


# ---------------------------------------------------------------------------
# Audio processing (identical to generate_audio.py)
# ---------------------------------------------------------------------------


def _process(data: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono, pad/trim to 5 s, peak-normalize."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        data = librosa.resample(
            data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
        )
    if len(data) > N_SAMPLES:
        data = data[:N_SAMPLES]
    elif len(data) < N_SAMPLES:
        data = np.pad(data, (0, N_SAMPLES - len(data)))
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak * 0.95
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# ESC-50 helpers
# ---------------------------------------------------------------------------


def _fetch_esc50_meta() -> list[dict]:
    """Download and parse ESC-50 metadata CSV."""
    print("Fetching ESC-50 metadata...")
    r = requests.get(ESC50_META_URL, timeout=30)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    return list(reader)


def _select_files(meta: list[dict], target: int, n: int) -> list[str]:
    """Pick n files with different src_file (source recording) for a target."""
    candidates = [row for row in meta if int(row["target"]) == target]
    seen_src: set[str] = set()
    selected: list[str] = []
    for row in candidates:
        src = row["src_file"]
        if src not in seen_src:
            seen_src.add(src)
            selected.append(row["filename"])
            if len(selected) >= n:
                break
    return selected


def _download_esc50(filename: str) -> np.ndarray | None:
    """Download a single ESC-50 WAV and return processed waveform."""
    url = f"{ESC50_AUDIO_BASE}/{filename}"
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        data, sr = sf.read(io.BytesIO(r.content))
        return _process(data, sr)
    except Exception as e:
        print(f"    FAILED {filename}: {e}")
        return None


# ---------------------------------------------------------------------------
# UrbanSound8K (gunshot)
# ---------------------------------------------------------------------------

# Preferred gunshot files (known to classify well), different fsIDs
PREFERRED_GUNSHOTS = ["135528-6-2-0.wav", "180937-6-0-0.wav", "17307-6-0-0.wav"]


def _download_gunshots(n: int, force: bool) -> list[tuple[int, np.ndarray]]:
    """Extract n gun_shot samples from UrbanSound8K parquet shard 02.

    Returns list of (index_1based, waveform) tuples.
    """
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("  ERROR: install huggingface_hub and pyarrow")
        return []

    try:
        print("  Downloading UrbanSound8K shard 02...")
        path = hf_hub_download(
            "danavery/urbansound8K",
            "data/train-00002-of-00016-887e0748205b6fa9.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(path)
        df = table.to_pandas()
        guns = df[df["classID"] == 6]

        # Try preferred files first, then fill with others (different fsID)
        seen_fs: set[str] = set()
        results: list[tuple[int, np.ndarray]] = []
        idx = 1

        # Pass 1: preferred
        for pref_name in PREFERRED_GUNSHOTS:
            if idx > n:
                break
            dest = SOUNDS_DIR / "gunshot" / f"{idx}.wav"
            if dest.exists() and not force:
                idx += 1
                continue
            match = guns[guns["slice_file_name"] == pref_name]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            fs_id = pref_name.split("-")[0]
            seen_fs.add(fs_id)
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            results.append((idx, _process(data, sr)))
            print(f"    got {pref_name}")
            idx += 1

        # Pass 2: fill remaining from other fsIDs
        if idx <= n:
            for _, row in guns.iterrows():
                if idx > n:
                    break
                dest = SOUNDS_DIR / "gunshot" / f"{idx}.wav"
                if dest.exists() and not force:
                    idx += 1
                    continue
                fs_id = row["slice_file_name"].split("-")[0]
                if fs_id in seen_fs:
                    continue
                seen_fs.add(fs_id)
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
                results.append((idx, _process(data, sr)))
                print(f"    got {row['slice_file_name']}")
                idx += 1

        return results
    except Exception as e:
        print(f"  UrbanSound8K failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    force = "--force" in sys.argv
    print("Downloading demo sounds (5 classes × 3 files)...\n")

    # --- ESC-50 classes ---
    meta = _fetch_esc50_meta()

    for scenario, target_id in ESC50_TARGETS.items():
        scenario_dir = SOUNDS_DIR / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{scenario}] ESC-50 target={target_id}")

        filenames = _select_files(meta, target_id, N_PER_CLASS)
        for i, fname in enumerate(filenames, start=1):
            dest = scenario_dir / f"{i}.wav"
            if dest.exists() and not force:
                print(f"  {i}.wav exists, skipping")
                continue
            print(f"  {fname}...", end=" ", flush=True)
            waveform = _download_esc50(fname)
            if waveform is not None:
                sf.write(str(dest), waveform, SAMPLE_RATE)
                print(f"OK ({dest.stat().st_size // 1024} KB)")

    # --- Gunshot from UrbanSound8K ---
    scenario_dir = SOUNDS_DIR / "gunshot"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[gunshot] UrbanSound8K classID=6")

    all_exist = all(
        (scenario_dir / f"{i}.wav").exists() for i in range(1, N_PER_CLASS + 1)
    )
    if all_exist and not force:
        print("  all files exist, skipping")
    else:
        for idx, waveform in _download_gunshots(N_PER_CLASS, force):
            dest = scenario_dir / f"{idx}.wav"
            sf.write(str(dest), waveform, SAMPLE_RATE)
            print(f"  saved {idx}.wav ({dest.stat().st_size // 1024} KB)")

    # --- Summary ---
    total = sum(1 for _ in SOUNDS_DIR.rglob("*.wav"))
    print(f"\nDone: {total} sound files in {SOUNDS_DIR}")


if __name__ == "__main__":
    main()
