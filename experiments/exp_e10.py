"""E10 — few-shot prototype for special species via Perch embeddings + k-NN.

Goal
----
Demonstrate few-shot recognition of the pilot's flagship rare species
(соня / выхухоль — edible dormouse / Russian desman) without any training:
embed a handful of xeno-canto reference recordings per species with Perch,
then classify held-out clips by leave-one-out cosine k-NN (k=1) over the
normalized prototype embeddings. Metric: LOO accuracy.

Expected outcome tonight: SKIP. xeno-canto needs an API key/download the
nightly run does not have, and these species are absent from the open
bird-detection datasets. The script is wired so that the moment data lands
under datasets/xeno_canto/<species>/*.wav it runs unchanged.

Contract
--------
run(cfg) -> {"model","dataset","metric","value","notes"} | {"skip": reason}.
Needs >=2 species with >=3 clips each under datasets/{xeno_canto,...}.
Wrapper: experiments.wrappers.perch (reused from E2). Any of {data missing,
key missing, wrapper missing} -> graceful skip with the cause.
"""

from __future__ import annotations

import logging
from pathlib import Path

from experiments import common

logger = logging.getLogger("faun.experiments.e10")

MODEL = "Perch+kNN_fewshot"
METRIC = "loo_accuracy"
DEFAULT_SEED = 1337
DEFAULT_PER_CLASS = 20
TARGET_SR = 32000
WIN_S = 5.0
SPECIES_DIRS = ("xeno_canto", "xenocanto", "special_species", "rare_species")
MIN_PER_CLASS = 3


def _discover_species(cfg):
    base = Path(cfg["datasets"])
    for cand in SPECIES_DIRS:
        root = base / cand
        if root.is_dir():
            species = {}
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                clips = [
                    p
                    for p in sorted(sub.rglob("*"))
                    if p.suffix.lower() in common.AUDIO_EXTS
                ]
                if clips:
                    species[sub.name] = clips
            if species:
                return species, root.name
    return {}, None


def run(cfg: dict) -> dict:
    import numpy as np

    seed = int(cfg.get("seed", DEFAULT_SEED))
    per_class = int(cfg.get("n_samples", DEFAULT_PER_CLASS))

    species, root_name = _discover_species(cfg)
    if not species:
        return {
            "skip": "no xeno-canto / special-species data "
            "(expected: needs xeno-canto key/download not present "
            "in nightly run)"
        }
    usable = {s: c for s, c in species.items() if len(c) >= MIN_PER_CLASS}
    if len(usable) < 2:
        counts = {s: len(c) for s, c in species.items()}
        return {"skip": f"need >=2 species with >={MIN_PER_CLASS} clips; got {counts}"}

    try:
        from experiments.wrappers import perch
    except Exception as exc:
        return {"skip": f"perch wrapper unavailable: {type(exc).__name__}: {exc}"}

    def clip_vec(path):
        wav, sr = common.load_audio(path, target_sr=TARGET_SR, mono=True)
        wins = common.windows(wav, sr, win_s=WIN_S)  # [n, 160000]
        embs, _logits = perch.embed(wins)
        embs = np.asarray(embs)
        if embs.size == 0:
            return None
        v = embs.mean(axis=0).ravel()
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    X, y, failures = [], [], 0
    for label, clips in usable.items():
        for path in common.sample_files(Path(clips[0]).parent, n=per_class, seed=seed):
            try:
                v = clip_vec(path)
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    logger.warning("embed failed %s: %s", path.name, exc)
                continue
            if v is not None:
                X.append(v)
                y.append(label)

    classes = sorted(set(y))
    if len(classes) < 2 or len(X) < 2 * MIN_PER_CLASS:
        return {
            "skip": f"insufficient embeddings ({len(X)}) "
            f"across {len(classes)} species ({failures} failures)"
        }

    Xa = np.stack(X)
    ya = np.array(y)
    correct = 0
    for i in range(len(Xa)):
        sims = Xa @ Xa[i]
        sims[i] = -np.inf
        correct += int(ya[int(np.argmax(sims))] == ya[i])
    acc = correct / len(Xa)

    notes = (
        f"species={classes} N={len(Xa)} dim={Xa.shape[1]} "
        f"k=1-cosine-LOO seed={seed} failures={failures}; few-shot prototype"
    )
    return {
        "model": MODEL,
        "dataset": root_name,
        "metric": METRIC,
        "value": round(acc, 4),
        "notes": notes,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from experiments.runner import build_cfg

    print(run(build_cfg(os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data"))))
