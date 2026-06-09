"""Tests for loop.nightgate — deterministic pre-gate for the autonomous tuning loop.

Tests the pure should_run() function first (all cases including new signature),
then main()-level integration tests. Also tests the heartbeat-once-per-loop-day
behaviour introduced by Fix #3.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import nightgate
from loop import state as state_mod


# ---------------------------------------------------------------------------
# Pure function tests — should_run(is_daytime, last_tick_completed_day, current_loop_day)
# ---------------------------------------------------------------------------

def test_daytime_always_skips():
    """Daytime → skip regardless of completion state."""
    ok, reason = nightgate.should_run(
        is_daytime=True,
        last_tick_completed_day=None,
        current_loop_day="2026-06-08",
    )
    assert ok is False
    assert "daytime" in reason


def test_daytime_skips_even_when_date_is_old():
    """Daytime → skip even when last run was yesterday."""
    ok, reason = nightgate.should_run(
        is_daytime=True,
        last_tick_completed_day="2026-06-07",
        current_loop_day="2026-06-08",
    )
    assert ok is False
    assert "daytime" in reason


def test_night_no_prior_run_proceeds():
    """Night + no last_tick_completed_day → proceed (first-ever run)."""
    ok, reason = nightgate.should_run(
        is_daytime=False,
        last_tick_completed_day=None,
        current_loop_day="2026-06-08",
    )
    assert ok is True
    assert "proceed" in reason


def test_night_run_already_done_today_skips():
    """Night + last_tick_completed_day == current_loop_day → skip."""
    ok, reason = nightgate.should_run(
        is_daytime=False,
        last_tick_completed_day="2026-06-08",
        current_loop_day="2026-06-08",
    )
    assert ok is False
    assert "done" in reason
    assert "2026-06-08" in reason


def test_night_run_done_yesterday_proceeds():
    """Night + last_tick_completed_day is an earlier loop-day → proceed."""
    ok, reason = nightgate.should_run(
        is_daytime=False,
        last_tick_completed_day="2026-06-07",
        current_loop_day="2026-06-08",
    )
    assert ok is True
    assert "proceed" in reason


# ---------------------------------------------------------------------------
# Heartbeat tests (Fix #3) — once per loop-day on skip
# ---------------------------------------------------------------------------

def test_skip_sends_heartbeat_first_time(tmp_path, monkeypatch):
    """On a skip (daytime), nightgate sends a heartbeat if not yet sent today."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": None,
        "last_heartbeat_loopday": None,
    })

    send_calls = []
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda state_path, loop_day_str: send_calls.append(loop_day_str))
    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1
    assert send_calls == ["2026-06-08"], "heartbeat must be sent once on first skip"


def test_skip_does_not_resend_heartbeat_same_loop_day(tmp_path, monkeypatch):
    """On a second skip with the same loop-day, heartbeat is NOT resent."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": None,
        "last_heartbeat_loopday": "2026-06-08",  # already sent today
    })

    send_calls = []
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda state_path, loop_day_str: send_calls.append(loop_day_str))
    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1
    assert send_calls == [], "heartbeat must NOT be resent on same loop-day"


def test_heartbeat_send_failure_does_not_crash_gate(tmp_path, monkeypatch):
    """If the heartbeat send raises, the gate still exits 1 cleanly (best-effort)."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": None,
        "last_heartbeat_loopday": None,
    })

    def _failing_send(state_path, loop_day_str):
        raise RuntimeError("Telegram is down")

    monkeypatch.setattr(nightgate, "_send_heartbeat", _failing_send)
    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    # Must still exit 1 (skip) — not crash with an unhandled exception
    assert exc_info.value.code == 1


def test_skip_night_completed_today_sends_heartbeat_once(tmp_path, monkeypatch):
    """Skip because already done → heartbeat sent if not yet sent today."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": "2026-06-08",
        "last_heartbeat_loopday": None,
    })

    send_calls = []
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda state_path, loop_day_str: send_calls.append(loop_day_str))
    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1
    assert send_calls == ["2026-06-08"]


# ---------------------------------------------------------------------------
# main()-level integration tests via injectable --state path + monkeypatching
# ---------------------------------------------------------------------------

def test_main_exits_1_when_daytime(tmp_path, monkeypatch):
    """main() exits 1 and prints skip reason when is_daytime returns True."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07", "last_heartbeat_loopday": "2026-06-08"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    # Heartbeat already sent — no network call needed
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1


def test_main_exits_0_when_night_and_not_done(tmp_path, monkeypatch):
    """main() exits 0 when it's night and tonight's run is not yet done."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)

    try:
        nightgate.main(["--state", str(sp)])
    except SystemExit as e:
        assert e.code == 0


def test_main_exits_1_when_already_done_tonight(tmp_path, monkeypatch):
    """main() exits 1 when last_tick_completed_day equals current loop-day."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": "2026-06-08",
        "last_heartbeat_loopday": "2026-06-08",  # already sent
    })

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1


def test_main_exits_0_when_no_state_file(tmp_path, monkeypatch):
    """main() exits 0 when state.json is missing (first-ever run at night)."""
    sp = tmp_path / "state.json"
    # Do NOT create the file — load_state returns {} for missing files.

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)

    try:
        nightgate.main(["--state", str(sp)])
    except SystemExit as e:
        assert e.code == 0
