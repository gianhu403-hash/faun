"""E4 — sanity gallery over a sample of the 180 GB raw trap corpus.

Goal
----
A human-readable contact sheet for the morning review: for a few files from
EACH trap folder A1..A5 (whatever finished downloading; missing/empty folder
-> noted, skipped), render

  * a mel/STFT spectrogram PNG of the raw clip,
  * top-k BirdNET species (names + confidence) via the birdnet wrapper,
  * top-k Perch logit indices (the perch wrapper returns raw logits without a
    public label map, so we list the strongest class indices as activity
    evidence, not species names),
  * a denoised variant: spectral subtraction of a "buzz" profile. If a file
    whose name contains buzz/office/hum exists in raw180 it is the noise
    profile; otherwise the floor is estimated from the quietest frames.

Artifacts -> results/e4_gallery/ (PNGs + gallery.md table).

Metric reported: files_rendered (a coverage count, not an accuracy score) —
E4 is eyeballable evidence the 180 GB pipeline ingests real trap audio.

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
Wrappers experiments.wrappers.{birdnet,perch} optional: absence degrades the
gallery (spectrograms only) but never crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiments import common

logger = logging.getLogger("faun.experiments.e4")

MODEL = "BirdNET+Perch_gallery"
METRIC = "files_rendered"
TRAPS = ["A1", "A2", "A3", "A4", "A5"]
DEFAULT_PER_TRAP = 4
DEFAULT_SEED = 1337
TOPK = 3
SR = 16000
PERCH_SR = 32000
WIN_S = 5.0


def _stft(wav, sr):
    from scipy.signal import stft

    f, t, Z = stft(wav, fs=sr, nperseg=1024, noverlap=512)
    return f, t, Z


def _buzz_profile(cfg):
    """Mean magnitude profile of a buzz/office/hum file in raw180, if any."""
    import numpy as np

    raw = Path(cfg["raw180"])
    if not raw.is_dir():
        return None, None
    for p in sorted(raw.rglob("*")):
        if p.suffix.lower() in common.AUDIO_EXTS and any(
            tok in p.name.lower() for tok in ("buzz", "office", "hum")
        ):
            try:
                wav, _ = common.load_audio(p, target_sr=SR, mono=True)
                _f, _t, Z = _stft(wav, SR)
                return np.abs(Z).mean(axis=1), p.name
            except Exception as exc:
                logger.warning("buzz profile load failed %s: %s", p.name, exc)
    return None, None


def _quiet_floor(wav, sr):
    import numpy as np

    _f, _t, Z = _stft(wav, sr)
    mag = np.abs(Z)
    energy = mag.sum(axis=0)
    if energy.size == 0:
        return None
    quiet = np.argsort(energy)[: max(1, energy.size // 10)]
    return mag[:, quiet].mean(axis=1)


def _denoise(wav, sr, profile):
    import numpy as np
    from scipy.signal import istft

    _f, _t, Z = _stft(wav, sr)
    mag, phase = np.abs(Z), np.angle(Z)
    prof = (
        profile
        if (profile is not None and profile.shape[0] == mag.shape[0])
        else _quiet_floor(wav, sr)
    )
    if prof is None:
        prof = 0.0
    clean_mag = np.clip(mag - np.asarray(prof)[:, None], 0.0, None)
    _t2, clean = istft(
        clean_mag * np.exp(1j * phase), fs=sr, nperseg=1024, noverlap=512
    )
    return clean.astype("float32")


def _birdnet_topk(path):
    try:
        from experiments.wrappers import birdnet

        dets = birdnet.analyze_file(str(path), min_conf=0.05)
        best = sorted(dets, key=lambda d: -d[3])[:TOPK]
        return (
            ", ".join(f"{name}:{conf:.2f}" for _s, _e, name, conf in best) or "(none)"
        )
    except Exception as exc:
        logger.warning("birdnet topk failed %s: %s", Path(path).name, exc)
        return "n/a"


def _perch_topk(path):
    """Top-k Perch logit indices (no public label map -> indices, not names)."""
    import numpy as np

    try:
        from experiments.wrappers import perch

        wav, sr = common.load_audio(path, target_sr=PERCH_SR, mono=True)
        wins = common.windows(wav, sr, win_s=WIN_S)
        _embs, logits = perch.embed(wins)
        if logits is None:
            return "emb-only"
        mean_logits = np.asarray(logits).mean(axis=0)
        top = np.argsort(mean_logits)[::-1][:TOPK]
        return ", ".join(f"#{int(i)}:{float(mean_logits[i]):.2f}" for i in top)
    except Exception as exc:
        logger.warning("perch topk failed %s: %s", Path(path).name, exc)
        return "n/a"


def run(cfg: dict) -> dict:
    seed = int(cfg.get("seed", DEFAULT_SEED))
    per_trap = int(cfg.get("per_trap", DEFAULT_PER_TRAP))
    raw = Path(cfg["raw180"])
    if not raw.is_dir():
        return {"skip": "raw180/ directory absent"}

    out_dir = Path(cfg["results_dir"]) / "e4_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    bn_on = _wrapper_available("birdnet")
    perch_on = _wrapper_available("perch")
    buzz_profile, buzz_name = _buzz_profile(cfg)

    rows: list[str] = []
    rendered = 0
    missing: list[str] = []
    for trap in TRAPS:
        tdir = raw / trap
        if not tdir.is_dir():
            missing.append(trap)
            continue
        files = common.sample_files(tdir, n=per_trap, seed=seed)
        if not files:
            missing.append(f"{trap}(empty)")
            continue
        for path in files:
            try:
                wav, sr = common.load_audio(path, target_sr=SR, mono=True)
            except Exception as exc:
                logger.warning("load failed %s: %s", path.name, exc)
                continue
            raw_png = out_dir / f"{trap}_{path.stem}_raw.png"
            try:
                common.save_spectrogram_png(wav, sr, raw_png)
            except Exception as exc:
                logger.warning("spectrogram failed %s: %s", path.name, exc)
                continue
            rendered += 1

            den_cell = "(failed)"
            try:
                clean = _denoise(wav, sr, buzz_profile)
                den_png = out_dir / f"{trap}_{path.stem}_denoised.png"
                common.save_spectrogram_png(clean, sr, den_png)
                den_cell = f"![den]({den_png.name})"
            except Exception as exc:
                logger.warning("denoise failed %s: %s", path.name, exc)

            bn_cell = _birdnet_topk(path) if bn_on else "off"
            pe_cell = _perch_topk(path) if perch_on else "off"
            rows.append(
                f"| {trap} | {path.name} | ![raw]({raw_png.name}) | "
                f"{den_cell} | {bn_cell} | {pe_cell} |"
            )

    den_desc = (
        f"buzz file {buzz_name}" if buzz_name else "per-clip quiet-floor estimate"
    )
    md = [
        "# E4 sanity gallery — 180 GB trap corpus",
        "",
        f"Traps rendered: {sorted(set(TRAPS) - set(t.split('(')[0] for t in missing))}  ",
        f"Missing/empty (not yet downloaded): {missing or 'none'}  ",
        f"Denoise profile: {den_desc}  ",
        f"BirdNET: {'on' if bn_on else 'off'} · Perch: {'on' if perch_on else 'off'}",
        "",
        "| trap | file | raw | denoised | top-k BirdNET | top-k Perch (logit idx) |",
        "|---|---|---|---|---|---|",
        *rows,
    ]
    (out_dir / "gallery.md").write_text("\n".join(md))

    if rendered == 0:
        return {"skip": f"no renderable audio in traps; missing={missing}"}

    notes = (
        f"rendered={rendered} missing={missing} denoise="
        f"{'buzz:' + buzz_name if buzz_name else 'quiet-floor'} "
        f"birdnet={'on' if bn_on else 'off'} perch={'on' if perch_on else 'off'}; "
        f"gallery={out_dir.name}/gallery.md"
    )
    return {
        "model": MODEL,
        "dataset": "raw180",
        "metric": METRIC,
        "value": rendered,
        "notes": notes,
    }


def _wrapper_available(name: str) -> bool:
    try:
        __import__(f"experiments.wrappers.{name}")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from experiments.runner import build_cfg

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
