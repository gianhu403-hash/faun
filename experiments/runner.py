"""Раннер экспериментов.

Запуск:
    python -m experiments.runner E1 E3
    python -m experiments.runner --all
    python -m experiments.runner E1 --timeout-min 10 --data-root /home/oleg/faun-data

Каждый эксперимент — модуль experiments/exp_e<N>.py с функцией
    run(cfg: dict) -> dict
который возвращает {"metric": ..., "value": ..., "notes": ...,
опционально "model", "dataset"} или {"skip": "<причина>"} для graceful skip.

Эксперимент исполняется в subprocess (изоляция тяжёлых либ + жёсткий timeout).
Ошибка или таймаут одного эксперимента НЕ валит очередь: в results.csv пишется
строка со status error/timeout/skip, очередь продолжается.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from experiments import common

PKG_DIR = Path(__file__).resolve().parent
PKG_PARENT = PKG_DIR.parent

DEFAULT_DATA_ROOT = os.environ.get("FAUN_DATA_ROOT", "/home/oleg/faun-data")
DEFAULT_TIMEOUT_MIN = float(os.environ.get("FAUN_EXP_TIMEOUT_MIN", "45"))
DEFAULT_RESULTS = os.environ.get("FAUN_RESULTS_CSV", str(PKG_DIR / "results.csv"))
DEFAULT_LOGS_DIR = str(PKG_DIR / "logs")

VRAM_POLL_S = 5.0


def build_cfg(data_root: str) -> dict:
    """Конфиг, передаваемый в run(cfg) каждого эксперимента."""
    root = Path(data_root)
    return {
        "data_root": str(root),
        "raw180": str(root / "raw180"),
        "datasets": str(root / "datasets"),
        "hf_cache": str(root / "hf_cache"),
        "results_dir": str(root / "results"),
    }


def discover_experiments() -> list[str]:
    """Все exp_e*.py в пакете -> имена вида E1, отсортированные по номеру."""
    names = []
    for p in PKG_DIR.glob("exp_e*.py"):
        m = re.fullmatch(r"exp_e(\d+)", p.stem)
        if m:
            names.append((int(m.group(1)), f"E{m.group(1)}"))
    return [n for _, n in sorted(names)]


def query_vram_mb() -> int | None:
    """Текущая занятая VRAM в МБ (max по GPU) через nvidia-smi; None если недоступен."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        values = [int(s) for s in out.stdout.split() if s.strip().isdigit()]
        return max(values) if values else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


class _VramMonitor:
    """Фоновый поллинг nvidia-smi на время эксперимента; хранит пик."""

    def __init__(self):
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            v = query_vram_mb()
            if v is None:
                return  # nvidia-smi недоступен — больше не пробуем
            self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(VRAM_POLL_S)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def _tail(path: Path, n_chars: int = 300) -> str:
    try:
        return path.read_text(errors="replace")[-n_chars:].strip()
    except OSError:
        return ""


def run_one(
    name: str,
    cfg: dict,
    results_csv: str | Path,
    logs_dir: str | Path,
    timeout_s: float,
) -> dict:
    """Исполняет один эксперимент в subprocess и пишет строку в results.csv.

    Возвращает записанную строку как dict (для логов/тестов).
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"

    out_json = Path(tempfile.mkstemp(prefix=f"faun_{name}_", suffix=".json")[1])
    cmd = [
        sys.executable,
        "-m",
        "experiments.runner",
        "--child",
        name,
        "--out",
        str(out_json),
        "--data-root",
        cfg["data_root"],
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{PKG_PARENT}{os.pathsep}" + os.environ.get("PYTHONPATH", ""),
    }

    t0 = time.monotonic()
    timed_out = False
    with open(log_path, "w") as log, _VramMonitor() as vram:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(PKG_PARENT)
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    runtime_s = round(time.monotonic() - t0, 1)
    vram_mb = vram.peak if vram.peak is not None else ""

    row = {
        "model": name,
        "dataset": "",
        "metric": "status",
        "value": "error",
        "runtime_s": runtime_s,
        "vram_mb": vram_mb,
        "notes": "",
    }

    if timed_out:
        row["value"] = "timeout"
        row["notes"] = f"timeout after {int(timeout_s)}s; log: {log_path}"
    else:
        try:
            payload = json.loads(out_json.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {
                "ok": False,
                "error": f"no result json (exit={proc.returncode}); "
                f"log tail: {_tail(log_path)}",
            }
        if payload.get("ok"):
            result = payload["result"]
            if "skip" in result:
                row["value"] = "skip"
                row["notes"] = str(result["skip"])
            else:
                row.update(
                    model=result.get("model", name),
                    dataset=result.get("dataset", ""),
                    metric=result.get("metric", ""),
                    value=result.get("value", ""),
                    notes=result.get("notes", ""),
                )
        else:
            row["notes"] = str(payload.get("error", "unknown error"))

    out_json.unlink(missing_ok=True)
    common.append_result(results_csv, **row)
    return row


def child_main(name: str, out_path: str, data_root: str) -> int:
    """Дочерний процесс: импорт exp-модуля, run(cfg), JSON-результат в out_path."""
    payload: dict
    try:
        module = importlib.import_module(f"experiments.exp_{name.lower()}")
        result = module.run(build_cfg(data_root))
        if not isinstance(result, dict):
            raise TypeError(f"run() must return dict, got {type(result).__name__}")
        payload = {"ok": True, "result": result}
    except Exception as e:
        import traceback

        payload = {
            "ok": False,
            "error": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=20)}",
        }
    Path(out_path).write_text(json.dumps(payload, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.runner", description=__doc__)
    ap.add_argument("names", nargs="*", help="Имена экспериментов: E1 E3 ...")
    ap.add_argument("--all", action="store_true", help="Все exp_e*.py по порядку")
    ap.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    # внутренний child-режим
    ap.add_argument("--child", metavar="NAME", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:
        return child_main(args.child, args.out, args.data_root)

    names = discover_experiments() if args.all else [n.upper() for n in args.names]
    if not names:
        ap.error("укажите эксперименты (E1 E3 ...) или --all")

    cfg = build_cfg(args.data_root)
    common.ensure_results_csv(args.results)
    print(
        f"results: {args.results} | data_root: {cfg['data_root']} "
        f"| timeout: {args.timeout_min} min"
    )

    for name in names:
        print(f"[{name}] running ...", flush=True)
        row = run_one(
            name, cfg, args.results, args.logs_dir, timeout_s=args.timeout_min * 60
        )
        print(
            f"[{name}] {row['metric']}={row['value']} "
            f"({row['runtime_s']}s) {row['notes'][:120]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
