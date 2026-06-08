"""Tests for src/loop/metrics.py — Wilson CI, FN unmeasured, idempotent CSV."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import metrics


def test_wilson_ci_known_value():
    # 40 successes of 100 at 95%: classic Wilson interval ≈ (0.307, 0.500).
    low, high = metrics.wilson_ci(40, 100)
    assert abs(low - 0.3066) < 0.005
    assert abs(high - 0.5000) < 0.005
    assert low < 0.40 < high


def test_wilson_ci_zero_trials_is_full_interval():
    low, high = metrics.wilson_ci(0, 0)
    assert low == 0.0
    assert high == 1.0


def test_wilson_ci_all_successes():
    low, high = metrics.wilson_ci(10, 10)
    assert high == 1.0
    assert 0.0 < low < 1.0


def test_compute_metrics_fp_from_labeled_rows():
    # 3 of 5 labeled triggers are false_positive → fp_rate 0.6.
    rows = [
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "animal"},
        {"reconciled_label": "animal"},
        {"reconciled_label": None},  # unlabeled → excluded from FP denominator
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["labeled_triggers"] == 5
    assert abs(m["fp_rate"] - 0.6) < 1e-9
    assert m["fp_ci"][0] < 0.6 < m["fp_ci"][1]


def test_compute_metrics_fn_unmeasured_when_no_audit():
    rows = [{"reconciled_label": "animal"}]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["fn_rate"] == "unmeasured"
    assert m["fn_ci"] is None


def test_compute_metrics_fn_measured_when_audit_present():
    rows = [{"reconciled_label": "animal"}]
    # 2 missed animals of 8 animal-present frames → fn_rate 0.25.
    m = metrics.compute_metrics(rows, fn_audit={"missed": 2, "animal_frames": 8})
    assert abs(m["fn_rate"] - 0.25) < 1e-9
    assert m["fn_ci"] is not None


def test_daily_csv_appends_one_row_per_date(tmp_path):
    csv_path = tmp_path / "daily.csv"
    m = {"labeled_triggers": 5, "total_triggers": 6, "fp_count": 3,
         "fp_rate": 0.6, "fp_ci": (0.3, 0.8), "fn_rate": "unmeasured", "fn_ci": None}
    metrics.append_daily(csv_path, "2026-06-10", m)
    metrics.append_daily(csv_path, "2026-06-11", m)
    with open(csv_path) as f:
        data_rows = list(csv.DictReader(f))
    assert len(data_rows) == 2
    assert {r["date"] for r in data_rows} == {"2026-06-10", "2026-06-11"}


def test_daily_csv_rerun_same_date_overwrites(tmp_path):
    csv_path = tmp_path / "daily.csv"
    m1 = {"labeled_triggers": 5, "total_triggers": 6, "fp_count": 3,
          "fp_rate": 0.6, "fp_ci": (0.3, 0.8), "fn_rate": "unmeasured", "fn_ci": None}
    m2 = {"labeled_triggers": 10, "total_triggers": 11, "fp_count": 2,
          "fp_rate": 0.2, "fp_ci": (0.05, 0.5), "fn_rate": "unmeasured", "fn_ci": None}
    metrics.append_daily(csv_path, "2026-06-10", m1)
    metrics.append_daily(csv_path, "2026-06-10", m2)  # same date re-run
    with open(csv_path) as f:
        data_rows = list(csv.DictReader(f))
    assert len(data_rows) == 1
    assert data_rows[0]["fp_rate"] == "0.2"
