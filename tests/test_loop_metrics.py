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
    assert "images captured" in text


def test_metrics_main_last_metrics_readable_by_report_without_flatten(tmp_path, monkeypatch):
    """End-to-end: after metrics writes state (with real data), report reads
    state WITHOUT the caller manually flattening metrics.  render_summary must
    succeed on the raw state value (after the JSON list→tuple coercion report
    already does)."""
    import sys
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
    db = DatabaseManager(cfg)
    # Insert a real detection so this is a data-tick, not a no-data tick.
    db.log_detection(
        image_path="/x.jpg", motion_area=500,
        animals_detected=True, detection_count=1,
        gate_would_suppress=False,
    )

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
    assert "images captured" in text


# ---------------------------------------------------------------------------
# Bug fix: no-data tick must not clobber baseline or write degenerate CSV row
# ---------------------------------------------------------------------------

def _make_db_with_detections(tmp_path, monkeypatch, count=1):
    """Helper: create a temp DB, insert `count` detections, return (db, db_file)."""
    db_file = tmp_path / "detections.db"
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    from config import Config
    from database_manager import DatabaseManager
    cfg = Config.create_test_config()
    db = DatabaseManager(cfg)
    for _ in range(count):
        db.log_detection(
            image_path="/x.jpg", motion_area=500,
            animals_detected=False, detection_count=0,
            gate_would_suppress=True,
        )
    return db, db_file


def test_no_data_tick_preserves_baseline(tmp_path, monkeypatch):
    """When the watermark has already caught up to the latest detection id and
    no new detections have arrived, main() must:
      - NOT overwrite state.json["last_metrics"] (baseline preserved), and
      - NOT append a row to daily.csv, and
      - print JSON with status=="no_data".
    """
    import json
    import sys

    # First, build a DB with one detection and a state that has already
    # ingested it (watermark == max(id)).
    db, db_file = _make_db_with_detections(tmp_path, monkeypatch, count=1)

    # Determine the id that was assigned.
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    max_id = conn.execute("SELECT MAX(id) FROM detections").fetchone()[0]
    conn.close()

    # Fake a real baseline already in state.json — this is the value that must survive.
    baseline_metrics = {
        "date": "2026-06-08",
        "total_triggers": 84,
        "labeled_triggers": 67,
        "fp_count": 53,
        "fp_rate": 0.7910447761194029,
        "fp_ci": [0.674, 0.877],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    state_path = tmp_path / "state.json"
    csv_path = tmp_path / "daily.csv"
    from loop import state as state_mod
    state_mod.save_state(state_path, {
        "watermark": max_id,   # already at max — no new data
        "paused": False,
        "last_metrics": baseline_metrics,
    })

    # Capture stdout to inspect the no_data signal.
    import builtins
    original_print = builtins.print
    printed_lines = []
    def capturing_print(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        printed_lines.append(line)
        original_print(*args, **kwargs)
    monkeypatch.setattr(builtins, "print", capturing_print)

    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics",
         "--state", str(state_path),
         "--csv", str(csv_path),
         "--date", "2026-06-09"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    # 1. last_metrics must be unchanged (baseline preserved).
    updated = state_mod.load_state(state_path)
    assert updated.get("last_metrics") == baseline_metrics, (
        "no-data tick must NOT overwrite last_metrics"
    )

    # 2. daily.csv must not have been created / must have no rows.
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 0, (
            f"no-data tick must NOT append to daily.csv, found rows: {rows}"
        )

    # 3. Printed output must carry the no_data signal.
    assert printed_lines, "main() must print something on a no-data tick"
    output = json.loads(printed_lines[-1])
    assert output.get("status") == "no_data", (
        f"expected status=no_data, got: {output}"
    )
    assert output.get("baseline_preserved") is True


def test_data_tick_writes_metrics_and_csv(tmp_path, monkeypatch):
    """When new detections exist beyond the watermark, main() must write
    last_metrics and append a row to daily.csv (existing behavior preserved).
    """
    import sys

    db, db_file = _make_db_with_detections(tmp_path, monkeypatch, count=3)

    state_path = tmp_path / "state.json"
    csv_path = tmp_path / "daily.csv"
    from loop import state as state_mod
    # Watermark at 0 means all 3 detections are new.
    state_mod.save_state(state_path, {"watermark": 0, "paused": False})

    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics",
         "--state", str(state_path),
         "--csv", str(csv_path),
         "--date", "2026-06-09"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    updated = state_mod.load_state(state_path)
    lm = updated.get("last_metrics")
    assert lm is not None, "data tick must write last_metrics"
    assert lm["total_triggers"] == 3, (
        f"expected 3 triggers, got {lm['total_triggers']}"
    )

    assert csv_path.exists(), "data tick must create daily.csv"
    import csv as csv_mod
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-06-09"
    assert rows[0]["total_triggers"] == "3"


# ---------------------------------------------------------------------------
# Fix #2: error_count in daily.csv + fp_trustworthy
# ---------------------------------------------------------------------------

def test_compute_metrics_includes_error_count():
    """compute_metrics must return error_count for rows with detection_status='error'."""
    rows = [
        {"reconciled_label": "false_positive", "detection_status": "no_animal"},
        {"reconciled_label": None, "detection_status": "error"},
        {"reconciled_label": None, "detection_status": "error"},
        {"reconciled_label": "animal", "detection_status": "identified"},
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["error_count"] == 2


def test_compute_metrics_fp_trustworthy_true_when_low_error_rate():
    """fp_trustworthy is True when error_rate <= ERROR_RATE_UNTRUSTWORTHY_THRESHOLD."""
    # 0 errors out of 10 → trustworthy
    rows = [{"reconciled_label": "animal", "detection_status": "identified"}] * 10
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["fp_trustworthy"] is True
    assert m["error_rate"] == 0.0


def test_compute_metrics_fp_trustworthy_false_when_high_error_rate():
    """fp_trustworthy is False when error_rate > ERROR_RATE_UNTRUSTWORTHY_THRESHOLD (0.2)."""
    # 3 errors out of 10 → 30% > 20% threshold → untrustworthy
    rows = (
        [{"reconciled_label": "animal", "detection_status": "identified"}] * 7
        + [{"reconciled_label": None, "detection_status": "error"}] * 3
    )
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["fp_trustworthy"] is False
    assert abs(m["error_rate"] - 0.3) < 1e-9


def test_compute_metrics_fp_trustworthy_boundary_exactly_at_threshold():
    """error_rate == threshold (0.2 exactly) is trustworthy (<=)."""
    # 2 errors out of 10 → exactly 20%
    rows = (
        [{"reconciled_label": "animal", "detection_status": "identified"}] * 8
        + [{"reconciled_label": None, "detection_status": "error"}] * 2
    )
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["fp_trustworthy"] is True


def test_compute_metrics_zero_triggers_trustworthy():
    """Zero total_triggers → error_rate 0.0, fp_trustworthy True (guard divide-by-zero)."""
    m = metrics.compute_metrics([], fn_audit=None)
    assert m["error_rate"] == 0.0
    assert m["fp_trustworthy"] is True


def test_daily_csv_contains_error_count_column(tmp_path):
    """error_count must appear as a column in daily.csv."""
    import csv as csv_mod
    csv_path = tmp_path / "daily.csv"
    m = {
        "labeled_triggers": 5, "total_triggers": 6, "fp_count": 3,
        "fp_rate": 0.6, "fp_ci": (0.3, 0.8), "fn_rate": "unmeasured", "fn_ci": None,
        "error_count": 1,
    }
    metrics.append_daily(csv_path, "2026-06-10", m)
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert "error_count" in fieldnames, f"error_count not in CSV fields: {fieldnames}"
    assert rows[0]["error_count"] == "1"


def test_daily_csv_old_rows_parseable_without_error_count(tmp_path):
    """Old rows missing error_count column are still parseable after migration."""
    import csv as csv_mod
    csv_path = tmp_path / "daily.csv"
    # Write a legacy CSV without error_count.
    old_fields = [
        "date", "total_triggers", "labeled_triggers", "fp_count", "fp_rate",
        "fp_ci_low", "fp_ci_high", "fn_rate", "fn_ci_low", "fn_ci_high",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({
            "date": "2026-06-01", "total_triggers": "10", "labeled_triggers": "8",
            "fp_count": "3", "fp_rate": "0.375", "fp_ci_low": "0.1", "fp_ci_high": "0.7",
            "fn_rate": "unmeasured", "fn_ci_low": "", "fn_ci_high": "",
        })
    # Appending a new row with error_count should not crash.
    m = {
        "labeled_triggers": 5, "total_triggers": 6, "fp_count": 2,
        "fp_rate": 0.4, "fp_ci": (0.1, 0.7), "fn_rate": "unmeasured", "fn_ci": None,
        "error_count": 0,
    }
    metrics.append_daily(csv_path, "2026-06-10", m)
    # Both rows must be readable.
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    dates = {r["date"] for r in rows}
    assert dates == {"2026-06-01", "2026-06-10"}


# ---------------------------------------------------------------------------
# Fix #5: metrics.main() persists watermark to state.json
# ---------------------------------------------------------------------------

def test_metrics_main_advances_watermark_on_data_tick(tmp_path, monkeypatch):
    """On a data tick, main() must advance state['watermark'] to new_watermark."""
    import sys
    db, db_file = _make_db_with_detections(tmp_path, monkeypatch, count=2)

    state_path = tmp_path / "state.json"
    csv_path = tmp_path / "daily.csv"
    from loop import state as state_mod
    state_mod.save_state(state_path, {"watermark": 0, "paused": False})

    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics", "--state", str(state_path),
         "--csv", str(csv_path), "--date", "2026-06-09"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    updated = state_mod.load_state(state_path)
    # Watermark must have advanced beyond 0 (to the max detection id).
    assert updated["watermark"] > 0, (
        f"watermark must be advanced after data tick, got {updated['watermark']}"
    )


def test_metrics_main_watermark_no_op_on_no_data_tick(tmp_path, monkeypatch):
    """On a no-data tick, watermark write is a harmless no-op (same value)."""
    import sys
    import sqlite3
    db, db_file = _make_db_with_detections(tmp_path, monkeypatch, count=1)

    # Set watermark to max id so there's no new data.
    conn = sqlite3.connect(str(db_file))
    max_id = conn.execute("SELECT MAX(id) FROM detections").fetchone()[0]
    conn.close()

    state_path = tmp_path / "state.json"
    csv_path = tmp_path / "daily.csv"
    from loop import state as state_mod
    state_mod.save_state(state_path, {"watermark": max_id, "paused": False})

    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics", "--state", str(state_path),
         "--csv", str(csv_path), "--date", "2026-06-09"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    updated = state_mod.load_state(state_path)
    # Watermark is unchanged (same as max_id, no new data).
    assert updated["watermark"] == max_id


# ---------------------------------------------------------------------------
# Per-tier FP partition: human / Claude (tier2) / MegaDetector (tier1)
# ---------------------------------------------------------------------------

def _row(human=None, tier2=None, tier1=None):
    """Build a minimal ingest row with the tier-label fields."""
    reconciled = human or tier2 or tier1
    return {
        "human": human,
        "tier2": tier2,
        "tier1": tier1,
        "reconciled_label": reconciled,
        "detection_status": "no_animal",
    }


def test_per_tier_partition_no_overlap_sums_to_labeled():
    """n_human + n_claude + n_md must equal labeled_triggers (no overlap, no gap)."""
    rows = [
        _row(human="false_positive"),
        _row(human="animal"),
        _row(tier2="false_positive"),
        _row(tier2="animal"),
        _row(tier1="false_positive"),
        _row(tier1="no_animal"),
        {"reconciled_label": None, "human": None, "tier2": None, "tier1": None,
         "detection_status": "no_animal"},  # unlabeled — excluded
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["n_human"] + m["n_claude"] + m["n_md"] == m["labeled_triggers"]
    assert m["labeled_triggers"] == 6


def test_per_tier_precedence_human_wins():
    """A row with both human and tier2 set lands in the human bucket only."""
    rows = [_row(human="animal", tier2="false_positive")]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["n_human"] == 1
    assert m["n_claude"] == 0
    assert m["fp_human_count"] == 0   # human label is "animal", not FP
    assert m["fp_claude_count"] == 0


def test_per_tier_precedence_claude_over_md():
    """A row with tier2 and tier1 (no human) → Claude bucket, not MD."""
    rows = [_row(tier2="false_positive", tier1="animal")]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["n_claude"] == 1
    assert m["n_md"] == 0
    assert m["fp_claude_count"] == 1
    assert m["fp_md_count"] == 0


def test_per_tier_fp_uses_own_tier_label():
    """Row where human='animal' but tier1='false_positive' → human-bucket non-FP (tier1 ignored)."""
    rows = [_row(human="animal", tier1="false_positive")]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["n_human"] == 1
    assert m["fp_human_count"] == 0  # own-tier label is "animal"
    assert m["n_md"] == 0            # tier1 ignored when human wins


def test_per_tier_artifact_regression():
    """5 human rows (1 FP) + 37 MD-only rows (35 FP): per-tier rates expose the blending artifact."""
    human_rows = (
        [_row(human="false_positive")]
        + [_row(human="animal")] * 4
    )
    md_rows = (
        [_row(tier1="false_positive")] * 35
        + [_row(tier1="animal")] * 2
    )
    rows = human_rows + md_rows
    m = metrics.compute_metrics(rows, fn_audit=None)

    assert m["n_human"] == 5
    assert abs(m["fp_human_rate"] - 0.2) < 1e-9

    assert m["n_md"] == 37
    assert abs(m["fp_md_rate"] - 35 / 37) < 1e-9

    # Headline blended rate is ~0.857 — very different from fp_human_rate 0.2.
    assert abs(m["fp_rate"] - 36 / 42) < 1e-9


# ---------------------------------------------------------------------------
# Task 2: persist per-tier fields to state.json + daily.csv
# ---------------------------------------------------------------------------

def test_jsonable_serializes_per_tier_ci_to_lists():
    """_jsonable must listify fp_human_ci, fp_claude_ci, fp_md_ci into JSON-serializable 2-element lists."""
    import json as _json
    rows = [
        _row(human="false_positive"),
        _row(tier2="animal"),
        _row(tier1="false_positive"),
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    j = metrics._jsonable(m)
    for key in ("fp_human_ci", "fp_claude_ci", "fp_md_ci"):
        assert isinstance(j[key], list), f"{key} must be a list, got {type(j[key])}"
        assert len(j[key]) == 2, f"{key} must have 2 elements"
    # Must round-trip through JSON without error.
    _json.dumps(j)


def test_csv_has_per_tier_columns(tmp_path):
    """append_daily must write n_human, fp_human_count, fp_human_rate,
    n_claude, fp_claude_count, fp_claude_rate, n_md, fp_md_count, fp_md_rate."""
    csv_path = tmp_path / "daily.csv"
    rows = [
        _row(human="false_positive"),
        _row(human="animal"),
        _row(tier2="false_positive"),
        _row(tier1="animal"),
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    metrics.append_daily(csv_path, "2026-06-10", m)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data_rows = list(reader)
    expected_cols = [
        "n_human", "fp_human_count", "fp_human_rate",
        "n_claude", "fp_claude_count", "fp_claude_rate",
        "n_md", "fp_md_count", "fp_md_rate",
    ]
    for col in expected_cols:
        assert col in fieldnames, f"column {col!r} missing from CSV header: {fieldnames}"
    row = data_rows[0]
    assert row["n_human"] == "2"
    assert row["fp_human_count"] == "1"
    assert row["n_claude"] == "1"
    assert row["fp_claude_count"] == "1"
    assert row["n_md"] == "1"
    assert row["fp_md_count"] == "0"


def test_csv_append_backward_compat(tmp_path):
    """An old CSV (only original columns, no per-tier columns) survives append_daily:
    old rows get blank new columns, no crash."""
    csv_path = tmp_path / "daily.csv"
    # Write an old-style CSV with only the original 11 columns (no per-tier).
    old_fields = [
        "date", "total_triggers", "labeled_triggers", "fp_count", "fp_rate",
        "fp_ci_low", "fp_ci_high", "fn_rate", "fn_ci_low", "fn_ci_high",
        "error_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({
            "date": "2026-06-01", "total_triggers": "10", "labeled_triggers": "8",
            "fp_count": "3", "fp_rate": "0.375", "fp_ci_low": "0.1", "fp_ci_high": "0.7",
            "fn_rate": "unmeasured", "fn_ci_low": "", "fn_ci_high": "",
            "error_count": "2",
        })
    # Append a new row with per-tier data — must not crash.
    new_rows = [_row(human="false_positive"), _row(tier1="animal")]
    m = metrics.compute_metrics(new_rows, fn_audit=None)
    metrics.append_daily(csv_path, "2026-06-10", m)
    # Both rows readable; old row has blank per-tier columns.
    with open(csv_path) as f:
        data_rows = list(csv.DictReader(f))
    assert len(data_rows) == 2
    assert {r["date"] for r in data_rows} == {"2026-06-01", "2026-06-10"}
    old_row = next(r for r in data_rows if r["date"] == "2026-06-01")
    per_tier_cols = [
        "n_human", "fp_human_count", "fp_human_rate",
        "n_claude", "fp_claude_count", "fp_claude_rate",
        "n_md", "fp_md_count", "fp_md_rate",
    ]
    for col in per_tier_cols:
        assert old_row[col] == "", f"old row should have blank {col!r}, got {old_row[col]!r}"
