"""Smoke-тесты каркаса (без тяжёлых либ). pytest experiments/test_smoke.py"""

import csv

import numpy as np
import pytest

from experiments import common, runner


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_results_csv_created_with_header(tmp_path):
    path = tmp_path / "results.csv"
    common.ensure_results_csv(path)
    rows = _read_csv(path)
    assert rows == [common.RESULTS_HEADER]
    # повторный вызов не дублирует заголовок
    common.ensure_results_csv(path)
    assert _read_csv(path) == [common.RESULTS_HEADER]


def test_append_result_sanitizes_notes(tmp_path):
    path = tmp_path / "results.csv"
    common.append_result(path, model="m", metric="auc", value=0.9, notes="a\nb")
    rows = _read_csv(path)
    assert len(rows) == 2
    assert rows[1][0] == "m" and rows[1][2] == "auc"
    assert "\n" not in rows[1][6] and "a | b" == rows[1][6]


def test_windows_shapes():
    sr = 100
    x = np.arange(10 * sr, dtype=np.float32)  # 10 s
    w = common.windows(x, sr, win_s=2.0, hop_s=1.0)
    assert w.shape == (9, 200)
    assert np.array_equal(w[0], x[:200])
    assert np.array_equal(w[1], x[100:300])
    # без перекрытия: hop = win, неполный хвост отброшен
    w2 = common.windows(x[: 5 * sr + 30], sr, win_s=2.0)
    assert w2.shape == (2, 200)


def test_windows_pads_short_signal():
    w = common.windows(np.ones(50, dtype=np.float32), sr=100, win_s=1.0)
    assert w.shape == (1, 100)
    assert w[0, :50].sum() == 50 and w[0, 50:].sum() == 0


def test_metrics_manual():
    auc = common.auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert auc == pytest.approx(0.75)
    p, r = common.precision_recall([1, 1, 0, 0], [1, 0, 1, 0])
    assert (p, r) == (0.5, 0.5)


def test_runner_skip_path_writes_row(tmp_path):
    """E0 (шаблон) graceful-skip'ается: строка со status=skip, очередь не падает."""
    results = tmp_path / "results.csv"
    row = runner.run_one(
        "E0",
        runner.build_cfg(str(tmp_path / "no-data")),
        results_csv=results,
        logs_dir=tmp_path / "logs",
        timeout_s=60,
    )
    assert row["value"] == "skip"
    rows = _read_csv(results)
    assert rows[0] == common.RESULTS_HEADER
    assert rows[1][2] == "status" and rows[1][3] == "skip"
    assert "no data" in rows[1][6]
    assert (tmp_path / "logs" / "E0.log").exists()


def test_runner_error_path_writes_row_and_continues(tmp_path):
    """Несуществующий эксперимент -> строка error, исключение не пробрасывается."""
    results = tmp_path / "results.csv"
    row = runner.run_one(
        "E99",
        runner.build_cfg(str(tmp_path)),
        results_csv=results,
        logs_dir=tmp_path / "logs",
        timeout_s=60,
    )
    assert row["value"] == "error"
    rows = _read_csv(results)
    assert rows[1][3] == "error"
    assert "ModuleNotFoundError" in rows[1][6] or "No module" in rows[1][6]


def test_discover_experiments_finds_template():
    names = runner.discover_experiments()
    assert "E0" in names
    assert names == sorted(names, key=lambda n: int(n[1:]))
