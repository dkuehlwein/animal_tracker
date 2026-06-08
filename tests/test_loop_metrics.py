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


# ---------------------------------------------------------------------------
# Defect 1: metrics.main() must write last_metrics into state.json
# ---------------------------------------------------------------------------

def test_metrics_main_writes_last_metrics_into_state(tmp_path, monkeypatch):
    """metrics CLI must write a flat last_metrics dict into state.json so that
    report.main() can consume it without any manual reshaping."""
    import json
    import sys

    # Build a temp state file and a temp DB with one detection.
    state_path = tmp_path / "state.json"
    db_file = tmp_path / "detections.db"
    csv_path = tmp_path / "daily.csv"
    state_data = {"watermark": 0, "paused": False}
    from loop import state as state_mod
    state_mod.save_state(state_path, state_data)

    # Create a temp DB via DatabaseManager.
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    from config import Config
    from database_manager import DatabaseManager
    cfg = Config.create_test_config()
    db = DatabaseManager(cfg)
    db.log_detection(
        image_path="/x.jpg", motion_area=500,
        animals_detected=False, detection_count=0,
        gate_would_suppress=True,
    )

    # Run the metrics module's main() with patched argv.
    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics",
         "--state", str(state_path),
         "--csv", str(csv_path),
         "--date", "2026-06-10"],
    )
    # main() calls sys.exit(1) on failure; we want it to succeed.
    from loop import metrics as metrics_mod
    metrics_mod.main()

    # State file must now have last_metrics.
    updated = state_mod.load_state(state_path)
    lm = updated.get("last_metrics")
    assert lm is not None, "last_metrics was not written to state.json"

    # Shape must be FLAT (what render_summary expects):
    # top-level: date, fp_rate, fp_ci, fn_rate, fn_ci,
    #             fp_count, labeled_triggers, total_triggers.
    assert "date" in lm, "last_metrics is missing 'date'"
    assert "fp_ci" in lm, "last_metrics is missing 'fp_ci'"
    assert "fp_rate" in lm, "last_metrics is missing 'fp_rate'"
    assert "fn_rate" in lm, "last_metrics is missing 'fn_rate'"
    assert isinstance(lm["fp_ci"], list), "fp_ci must be a JSON list"

    # It must round-trip through report.render_summary without crashing.
    lm_for_render = dict(lm)
    lm_for_render["fp_ci"] = tuple(lm["fp_ci"])
    lm_for_render["fn_ci"] = tuple(lm["fn_ci"]) if lm["fn_ci"] else None
    from loop import report as report_mod
    text = report_mod.render_summary(lm_for_render, state={"paused": False}, active_experiment={})
    assert "FP" in text


def test_metrics_main_last_metrics_readable_by_report_without_flatten(tmp_path, monkeypatch):
    """End-to-end: after metrics writes state, report reads state WITHOUT the
    caller manually flattening metrics.  render_summary must succeed on the raw
    state value (after the JSON list→tuple coercion report already does)."""
    import json, sys
    state_path = tmp_path / "state.json"
    db_file = tmp_path / "detections.db"
    csv_path = tmp_path / "daily.csv"
    from loop import state as state_mod
    state_mod.save_state(state_path, {"watermark": 0, "paused": False})

    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    from config import Config
    from database_manager import DatabaseManager
    cfg = Config.create_test_config()
    DatabaseManager(cfg)  # creates schema

    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics", "--state", str(state_path),
         "--csv", str(csv_path), "--date", "2026-06-10"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    st = state_mod.load_state(state_path)
    lm = st["last_metrics"]
    # Simulate exactly what report.main() does:
    lm_copy = dict(lm)
    lm_copy["fp_ci"] = tuple(lm_copy["fp_ci"])
    lm_copy["fn_ci"] = tuple(lm_copy["fn_ci"]) if lm_copy["fn_ci"] else None
    from loop import report as report_mod
    # Must not raise KeyError.
    text = report_mod.render_summary(lm_copy, state=st, active_experiment={})
    assert "Wildlife loop" in text
