"""E0 — шаблон эксперимента (всегда skip; копируй как exp_e<N>.py).

Контракт: run(cfg) -> dict.
  Успех: {"model": ..., "dataset": ..., "metric": ..., "value": ..., "notes": ...}
  Graceful skip (нет данных/кред): {"skip": "<причина>"}
cfg: data_root, raw180, datasets, hf_cache, results_dir (см. runner.build_cfg).
"""

from pathlib import Path


def run(cfg: dict) -> dict:
    if not Path(cfg["raw180"]).is_dir():
        return {"skip": f"no data: {cfg['raw180']} missing (E0 is a template)"}
    return {"skip": "E0 is a template experiment, nothing to run"}
