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


# ---------------------------------------------------------------------------
# Defect 2: --no-send flag renders without calling Telegram
# ---------------------------------------------------------------------------

def test_report_main_no_send_renders_without_telegram(tmp_path, monkeypatch):
    """--no-send must print the rendered text and NOT call send()."""
    import json, sys

    state_path = tmp_path / "state.json"
    lm = {
        "date": "2026-06-10",
        "total_triggers": 10,
        "labeled_triggers": 8,
        "fp_count": 3,
        "fp_rate": 0.375,
        "fp_ci": [0.1, 0.65],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    from loop import state as state_mod
    state_mod.save_state(state_path, {
        "paused": False,
        "active_experiment_id": None,
        "backlog": [],
        "last_metrics": lm,
    })

    # Ensure send() is NOT called (it would fail without real credentials).
    send_called = []
    monkeypatch.setattr(report, "send", lambda text: send_called.append(text) or (_ for _ in ()).throw(AssertionError("send() called with --no-send")))

    # Capture stdout.
    captured = []
    monkeypatch.setattr(sys, "argv", [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--no-send",
    ])

    import io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        report.main()
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout

    assert send_called == [], "send() must not be called when --no-send is used"
    parsed = json.loads(output.strip())
    assert parsed["sent"] is False
    assert "rendered" in parsed
    assert "Wildlife loop" in parsed["rendered"]
    assert "FP" in parsed["rendered"]


# ---------------------------------------------------------------------------
# Fix #2: untrustworthy FP alert in summary
# ---------------------------------------------------------------------------

def _metrics_with_trust(fp_trustworthy: bool, error_count: int = 3, total: int = 10):
    return {
        "date": "2026-06-10",
        "total_triggers": total,
        "labeled_triggers": total - error_count,
        "fp_count": 2,
        "fp_rate": 0.2,
        "fp_ci": (0.05, 0.45),
        "fn_rate": "unmeasured",
        "fn_ci": None,
        "error_count": error_count,
        "error_rate": error_count / total if total else 0.0,
        "fp_trustworthy": fp_trustworthy,
    }


def test_summary_includes_untrustworthy_alert_when_fp_not_trustworthy():
    """When fp_trustworthy is False, summary must include a clear alert line."""
    text = report.render_summary(
        metrics=_metrics_with_trust(fp_trustworthy=False, error_count=3, total=10),
        state={"paused": False},
        active_experiment={},
    )
    assert "UNTRUSTWORTHY" in text or "untrustworthy" in text.lower()
    assert "error_rate" in text or "30%" in text or "3/10" in text


def test_summary_does_not_include_untrustworthy_alert_when_trustworthy():
    """When fp_trustworthy is True, no alert line must appear."""
    text = report.render_summary(
        metrics=_metrics_with_trust(fp_trustworthy=True, error_count=0, total=10),
        state={"paused": False},
        active_experiment={},
    )
    assert "UNTRUSTWORTHY" not in text
    # "untrustworthy" must not appear in the normal good-case output
    assert "untrustworthy" not in text.lower()


def test_summary_backward_compat_without_fp_trustworthy():
    """render_summary must not crash when fp_trustworthy is absent (old state)."""
    m = {
        "date": "2026-06-10",
        "total_triggers": 5,
        "labeled_triggers": 5,
        "fp_count": 1,
        "fp_rate": 0.2,
        "fp_ci": (0.05, 0.5),
        "fn_rate": "unmeasured",
        "fn_ci": None,
        # No fp_trustworthy, no error_rate, no error_count — old shape
    }
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    assert "FP" in text


def test_report_main_no_send_does_not_require_telegram_credentials(tmp_path, monkeypatch):
    """--no-send must not import or instantiate Config with Telegram tokens."""
    import json, sys

    state_path = tmp_path / "state.json"
    lm = {
        "date": "2026-06-10",
        "total_triggers": 5,
        "labeled_triggers": 4,
        "fp_count": 1,
        "fp_rate": 0.25,
        "fp_ci": [0.05, 0.57],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    from loop import state as state_mod
    state_mod.save_state(state_path, {
        "paused": False,
        "active_experiment_id": None,
        "backlog": [],
        "last_metrics": lm,
    })

    # Remove Telegram env vars to prove they aren't needed.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    monkeypatch.setattr(sys, "argv", [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--no-send",
    ])

    import io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        report.main()
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout

    parsed = json.loads(output.strip())
    assert parsed["sent"] is False
    assert "FP" in parsed["rendered"]
