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


# ---------------------------------------------------------------------------
# days_behind() — pure function
# ---------------------------------------------------------------------------

def test_days_behind_normal_gap():
    assert nightgate.days_behind("2026-08-04", "2026-08-31") == 27


def test_days_behind_zero_gap():
    assert nightgate.days_behind("2026-06-08", "2026-06-08") == 0


def test_days_behind_none_input():
    assert nightgate.days_behind(None, "2026-06-08") is None


def test_days_behind_unparseable_input():
    assert nightgate.days_behind("not-a-date", "2026-06-08") is None


# ---------------------------------------------------------------------------
# should_alert_staleness() — pure function
# ---------------------------------------------------------------------------

def test_staleness_two_days_behind_no_alert():
    """Exactly at the threshold (2 days behind) → no alert."""
    ok = nightgate.should_alert_staleness(
        last_tick_completed_day="2026-06-06",
        current_loop_day="2026-06-08",
        last_alert_loopday=None,
    )
    assert ok is False


def test_staleness_three_days_behind_alerts():
    """Strictly more than 2 days behind → alert."""
    ok = nightgate.should_alert_staleness(
        last_tick_completed_day="2026-06-05",
        current_loop_day="2026-06-08",
        last_alert_loopday=None,
    )
    assert ok is True


def test_staleness_already_alerted_this_loopday_no_second_alert():
    ok = nightgate.should_alert_staleness(
        last_tick_completed_day="2026-06-05",
        current_loop_day="2026-06-08",
        last_alert_loopday="2026-06-08",
    )
    assert ok is False


def test_staleness_missing_last_tick_completed_day_no_alert():
    ok = nightgate.should_alert_staleness(
        last_tick_completed_day=None,
        current_loop_day="2026-06-08",
        last_alert_loopday=None,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# should_alert_camera_down() — pure function
# ---------------------------------------------------------------------------

def test_camera_inactive_alerts():
    ok = nightgate.should_alert_camera_down(
        camera_active=False,
        current_loop_day="2026-06-08",
        last_alert_loopday=None,
    )
    assert ok is True


def test_camera_active_no_alert():
    ok = nightgate.should_alert_camera_down(
        camera_active=True,
        current_loop_day="2026-06-08",
        last_alert_loopday=None,
    )
    assert ok is False


def test_camera_already_alerted_this_loopday_no_second_alert():
    ok = nightgate.should_alert_camera_down(
        camera_active=False,
        current_loop_day="2026-06-08",
        last_alert_loopday="2026-06-08",
    )
    assert ok is False


# ---------------------------------------------------------------------------
# main()-level: dead-man's switch checks (staleness + camera liveness)
# ---------------------------------------------------------------------------

def test_main_staleness_alert_fires_on_proceed_path(tmp_path, monkeypatch):
    """Alerts fire even when the gate PROCEEDS — this is the whole point:
    the 27-night OAuth outage was on ticks that passed the gate."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": "2026-06-01",  # 7 days behind
    })

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0
    assert "last_staleness_alert_loopday" in alert_calls


def test_main_camera_alert_fires_on_proceed_path(tmp_path, monkeypatch):
    """Camera-down alert fires even when the gate PROCEEDS."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: False)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0
    assert "last_camera_alert_loopday" in alert_calls


def test_main_alerts_fire_on_skip_path_too(tmp_path, monkeypatch):
    """Alerts also fire on a SKIP tick (daytime), alongside the heartbeat."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-01"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: False)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1
    assert "last_staleness_alert_loopday" in alert_calls
    assert "last_camera_alert_loopday" in alert_calls


def test_main_staleness_alert_failure_does_not_change_exit_code_proceed(tmp_path, monkeypatch):
    """A failing _send_alert must not change main()'s exit code on the proceed path."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-01"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)

    def _failing_alert(*a):
        raise RuntimeError("Telegram is down")

    monkeypatch.setattr(nightgate, "_send_alert", _failing_alert)

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0


def test_main_camera_check_failure_does_not_change_exit_code_skip(tmp_path, monkeypatch):
    """An exception from _is_camera_active must not change main()'s exit code
    on the skip path, and must not raise out of main()."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-08"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)

    def _failing_camera_check():
        raise TimeoutError("systemctl timed out")

    monkeypatch.setattr(nightgate, "_is_camera_active", _failing_camera_check)
    monkeypatch.setattr(nightgate, "_send_alert", lambda *a: None)

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1


def test_main_does_not_resend_staleness_alert_same_loop_day(tmp_path, monkeypatch):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": "2026-06-01",
        "last_staleness_alert_loopday": "2026-06-08",
    })

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit):
        nightgate.main(["--state", str(sp)])
    assert alert_calls == []


def test_main_does_not_resend_camera_alert_same_loop_day(tmp_path, monkeypatch):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {
        "last_tick_completed_day": "2026-06-07",
        "last_camera_alert_loopday": "2026-06-08",
    })

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: False)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit):
        nightgate.main(["--state", str(sp)])
    assert alert_calls == []


# ---------------------------------------------------------------------------
# main()-level: scene-liveness check (exp #18)
# ---------------------------------------------------------------------------

def test_main_scene_alert_fires_on_proceed_path(tmp_path, monkeypatch):
    """A scene-change measurement past threshold triggers the scene alert,
    even on the PROCEED path — same shape as staleness/camera."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)
    monkeypatch.setattr(nightgate, "_get_image_dir", lambda: tmp_path)
    monkeypatch.setattr(
        nightgate,
        "_measure_scene_match",
        lambda image_dir, now: {
            "match_fraction": 0.0, "n_recent": 12, "n_baseline": 10,
            "n_matched": 0, "median_best": 0.05,
        },
    )

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0
    assert "last_scene_change_alert_loopday" in alert_calls


def test_main_no_scene_alert_when_scene_matches(tmp_path, monkeypatch):
    """A matching-scene measurement never triggers the scene alert."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)
    monkeypatch.setattr(nightgate, "_get_image_dir", lambda: tmp_path)
    monkeypatch.setattr(
        nightgate,
        "_measure_scene_match",
        lambda image_dir, now: {
            "match_fraction": 0.9, "n_recent": 12, "n_baseline": 10,
            "n_matched": 11, "median_best": 0.8,
        },
    )

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0
    assert alert_calls == []


def test_main_scene_check_failure_does_not_change_exit_code(tmp_path, monkeypatch):
    """An exception from _measure_scene_match must not change main()'s exit
    code and must not raise out of main() (failure isolation)."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_tick_completed_day": "2026-06-07"})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_loop_day", lambda: "2026-06-08")
    monkeypatch.setattr(nightgate, "_send_heartbeat", lambda *a: None)
    monkeypatch.setattr(nightgate, "_is_camera_active", lambda: True)
    monkeypatch.setattr(nightgate, "_get_image_dir", lambda: tmp_path)

    def _failing_measure(image_dir, now):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr(nightgate, "_measure_scene_match", _failing_measure)

    alert_calls = []
    monkeypatch.setattr(
        nightgate,
        "_send_alert",
        lambda state_path, loop_day_str, text, stamp_key: alert_calls.append(stamp_key),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 0
    assert alert_calls == []
