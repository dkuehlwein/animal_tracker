"""Tests for src/loop/report.py — summary + heartbeat text, paused suppression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import report


def _metrics(fp_rate=0.38, fn="unmeasured"):
    return {
        "date": "2026-06-10",
        "total_triggers": 42,
        "labeled_triggers": 30,
        "fp_count": 11,
        "fp_rate": fp_rate,
        "fp_ci": (0.30, 0.46),
        "fn_rate": fn,
        "fn_ci": None if fn == "unmeasured" else (0.05, 0.18),
    }


def test_summary_includes_fp_and_fn():
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": False},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "FP" in text
    assert "FN" in text
    assert "unmeasured" in text
    assert "38%" in text or "0.38" in text
    assert "notification-gate-live" in text


def test_summary_suppresses_tuning_lines_when_paused():
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": True},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "PAUSED" in text
    # When paused we do not advertise active tuning as if it's progressing.
    assert "tuning frozen" in text.lower() or "paused" in text.lower()


def test_heartbeat_text_mentions_alive_and_timestamp():
    text = report.render_heartbeat(last_tick_iso="2026-06-10T23:14:00")
    assert "alive" in text.lower()
    assert "2026-06-10T23:14:00" in text
