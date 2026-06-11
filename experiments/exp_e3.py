"""E3 — stage-1 detector showdown: faun.ml (onset+NDSI) vs CLAP zero-shot.

Goal
----
Compare two cheap stage-1 gates ("is this 10s window interesting / bird vs
wind-silence-noise") on M random 10-second windows of REAL trap recordings
(raw180); if raw180 is empty, fall back to freefield1010 clips.

  Detector A — faun.ml: onset.py transient trigger OR an NDSI lean toward the
               biophonic 2-11 kHz band (bird song). Pure DSP, CPU.
  Detector B — CLAP zero-shot: scores text prompts {bird song...} vs
               {wind, silence, noise, rain}; positive when the best bird
               prompt beats the best noise prompt. GPU (CLAP weights CC0).

Output
------
  * 2x2 agreement matrix (rows=A, cols=B) + agreement rate,
  * coverage of each detector (fraction of windows flagged positive),
  * M spectrogram PNGs annotated with both verdicts in results/e3_windows/
    for a morning human eyeball pass.

Metric reported: agreement_rate (over windows where both voted).

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
Wrapper experiments.wrappers.clap optional: if absent, A-only PNGs are still
written (review not blocked) and the result is a skip noting CLAP missing.
faun.ml.onset / faun.ml.ndsi imported directly (rsynced with experiments).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from experiments import common

logger = logging.getLogger("faun.experiments.e3")

MODEL = "onset+ndsi_vs_CLAP"
METRIC = "agreement_rate"
WIN_S = 10.0
DEFAULT_M = 30
DEFAULT_SEED = 1337
DSP_SR = 16000  # onset/ndsi reference rate
CLAP_SR = 48000  # CLAP wrapper native rate
NDSI_BIO_THRESH = -0.3  # NDSI >= this => biophony present (bird-ish)
BIRD_PROMPTS = ["bird song", "bird call", "birds chirping"]
NOISE_PROMPTS = ["wind", "silence", "background noise", "rain"]


def _pick_windows(cfg, m, seed):
    """Return (source, [(name, start_s, wav_dsp16k, wav_clap48k)]) for m windows."""
    import numpy as np

    raw = Path(cfg["raw180"])
    files = common.sample_files(raw, n=m * 3, seed=seed) if raw.is_dir() else []
    source = "raw180"
    if not files:
        ds = Path(cfg["datasets"]) / "freefield1010"
        if not ds.is_dir():
            ds = Path(cfg["datasets"]) / "ff1010bird"
        files = common.sample_files(ds, n=m * 3, seed=seed) if ds.is_dir() else []
        source = ds.name if files else "none"
    if not files:
        return source, []

    rng = random.Random(seed)
    out = []
    for path in files:
        if len(out) >= m:
            break
        try:
            wav16, _ = common.load_audio(path, target_sr=DSP_SR, mono=True)
        except Exception as exc:
            logger.warning("load failed %s: %s", path.name, exc)
            continue
        wins = common.windows(wav16, DSP_SR, win_s=WIN_S)  # [n, win]
        idx = rng.randrange(wins.shape[0])
        chunk16 = np.asarray(wins[idx], dtype=np.float32)
        start_s = idx * WIN_S
        # CLAP wants 48k; resample just this window
        try:
            import soxr

            chunk48 = soxr.resample(chunk16, DSP_SR, CLAP_SR).astype("float32")
        except Exception:
            chunk48 = None
        out.append((path.name, float(start_s), chunk16, chunk48))
    return source, out[:m]


def _detector_a(wav16, sr) -> bool:
    from faun.ml.ndsi import compute_ndsi
    from faun.ml.onset import detect_onset

    onset = detect_onset(wav16, sample_rate=sr).triggered
    ndsi = compute_ndsi(wav16, sr=sr).ndsi
    return bool(onset or ndsi >= NDSI_BIO_THRESH)


def _detector_b_batch(clap, wavs48):
    """CLAP verdicts for a list of 48k windows. Returns list[bool|None]."""
    import numpy as np

    usable = [(i, w) for i, w in enumerate(wavs48) if w is not None]
    out: list = [None] * len(wavs48)
    if not usable:
        return out
    prompts = BIRD_PROMPTS + NOISE_PROMPTS
    n_bird = len(BIRD_PROMPTS)
    batch = np.stack([w for _i, w in usable])
    logits = np.asarray(clap.score(batch, prompts))  # [N, P]
    for row, (i, _w) in zip(logits, usable):
        out[i] = bool(row[:n_bird].max() > row[n_bird:].max())
    return out


def run(cfg: dict) -> dict:
    import numpy as np

    seed = int(cfg.get("seed", DEFAULT_SEED))
    m = int(cfg.get("m_windows", DEFAULT_M))

    source, wins = _pick_windows(cfg, m, seed)
    if not wins:
        return {"skip": "no audio in raw180 nor freefield1010 fallback"}

    out_dir = Path(cfg["results_dir"]) / "e3_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    clap = None
    try:
        from experiments.wrappers import clap as clap_mod

        clap = clap_mod
    except Exception as exc:
        logger.warning("CLAP wrapper unavailable: %s", exc)

    a_votes = [_detector_a(w16, DSP_SR) for _n, _s, w16, _w48 in wins]

    b_votes: list = [None] * len(wins)
    if clap is not None:
        try:
            b_votes = _detector_b_batch(clap, [w48 for _n, _s, _w16, w48 in wins])
        except Exception as exc:
            logger.warning("CLAP scoring failed: %s", exc)

    # annotated spectrograms for morning review
    for i, (name, start_s, w16, _w48) in enumerate(wins):
        a = a_votes[i]
        b = b_votes[i]
        verdict = (
            f"A={'EVENT' if a else 'quiet'} "
            f"B={'bird' if b else ('noise' if b is False else 'n/a')}"
        )
        png = out_dir / f"e3_{i:02d}_{Path(name).stem}.png"
        try:
            common.save_spectrogram_png(w16, DSP_SR, png)
            _annotate(png, f"{name} @{start_s:.0f}s | {verdict}")
        except Exception as exc:
            logger.warning("spectrogram failed %s: %s", name, exc)

    cov_a = float(np.mean(a_votes)) if a_votes else 0.0
    paired = [(a, b) for a, b in zip(a_votes, b_votes) if b is not None]

    if not paired:
        reason = "CLAP unavailable" if clap is None else "no CLAP votes"
        return {
            "skip": f"{reason}; A-only review PNGs in {out_dir.name}/ "
            f"(M={len(wins)}, coverage_A={cov_a:.2f}, source={source})"
        }

    agree = float(np.mean([a == b for a, b in paired]))
    mat = [[0, 0], [0, 0]]  # rows=A(0/1), cols=B(0/1)
    for a, b in paired:
        mat[int(a)][int(b)] += 1
    cov_b = float(np.mean([b for _a, b in paired]))

    notes = (
        f"M={len(paired)}/{len(wins)} paired source={source} "
        f"matrix(A_rows,B_cols)={mat} coverage_A={cov_a:.2f} "
        f"coverage_B={cov_b:.2f}; review PNGs in {out_dir.name}/"
    )
    return {
        "model": MODEL,
        "dataset": source,
        "metric": METRIC,
        "value": round(agree, 4),
        "notes": notes,
    }


def _annotate(png_path: Path, text: str) -> None:
    """Overlay a caption onto an existing PNG (best-effort, Pillow optional)."""
    try:
        from PIL import Image, ImageDraw

        img = Image.open(png_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
        draw.text((4, 3), text[:110], fill=(255, 255, 255))
        img.save(png_path)
    except Exception:
        # Caption is a nicety; the PNG + CSV notes already carry verdicts.
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from experiments.runner import build_cfg

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
