"""Tests for loop.nightgate — deterministic pre-gate for the autonomous tuning loop.

Tests the pure should_run() function first (four cases), then a main()-level
integration test that exercises the full CLI path with injectable dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import nightgate
from loop import state as state_mod


# ---------------------------------------------------------------------------
# Pure function tests — should_run(is_daytime, last_metrics_date, today)
# ---------------------------------------------------------------------------

def test_daytime_always_skips():
    """Daytime → skip regardless of last_metrics_date."""
    ok, reason = nightgate.should_run(is_daytime=True, last_metrics_date=None, today="2026-06-09")
    assert ok is False
    assert "daytime" in reason


def test_daytime_skips_even_when_date_is_old():
    """Daytime → skip even when last run was yesterday."""
    ok, reason = nightgate.should_run(is_daytime=True, last_metrics_date="2026-06-08", today="2026-06-09")
    assert ok is False
    assert "daytime" in reason


def test_night_no_prior_run_proceeds():
    """Night + no last_metrics date → proceed (first-ever run)."""
    ok, reason = nightgate.should_run(is_daytime=False, last_metrics_date=None, today="2026-06-09")
    assert ok is True
    assert "proceed" in reason


def test_night_run_already_done_today_skips():
    """Night + last_metrics.date == today → skip (run already done)."""
    ok, reason = nightgate.should_run(is_daytime=False, last_metrics_date="2026-06-09", today="2026-06-09")
    assert ok is False
    assert "done" in reason
    assert "2026-06-09" in reason


def test_night_run_done_yesterday_proceeds():
    """Night + last_metrics.date is an earlier date → proceed."""
    ok, reason = nightgate.should_run(is_daytime=False, last_metrics_date="2026-06-08", today="2026-06-09")
    assert ok is True
    assert "proceed" in reason


# ---------------------------------------------------------------------------
# main()-level integration test via injectable --state path + monkeypatching
# ---------------------------------------------------------------------------

def test_main_exits_1_when_daytime(tmp_path, monkeypatch):
    """main() exits 1 and prints skip reason when is_daytime returns True."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_metrics": {"date": "2026-06-08"}})

    # Patch SunChecker.is_daytime to return True
    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: True)
    monkeypatch.setattr(nightgate, "_get_today", lambda: "2026-06-09")

    import pytest
    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1


def test_main_exits_0_when_night_and_not_done(tmp_path, monkeypatch):
    """main() exits 0 when it's night and tonight's run is not yet done."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_metrics": {"date": "2026-06-08"}})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_today", lambda: "2026-06-09")

    # Should NOT raise SystemExit (exits 0, which in Python means no exception or
    # SystemExit(0)); we allow both.
    import pytest
    try:
        nightgate.main(["--state", str(sp)])
    except SystemExit as e:
        assert e.code == 0


def test_main_exits_1_when_already_done_tonight(tmp_path, monkeypatch):
    """main() exits 1 when last_metrics.date equals today."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"last_metrics": {"date": "2026-06-09"}})

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_today", lambda: "2026-06-09")

    import pytest
    with pytest.raises(SystemExit) as exc_info:
        nightgate.main(["--state", str(sp)])
    assert exc_info.value.code == 1


def test_main_exits_0_when_no_state_file(tmp_path, monkeypatch):
    """main() exits 0 when state.json is missing (first-ever run at night)."""
    sp = tmp_path / "state.json"
    # Do NOT create the file — load_state returns {} for missing files.

    monkeypatch.setattr(nightgate, "_get_is_daytime", lambda: False)
    monkeypatch.setattr(nightgate, "_get_today", lambda: "2026-06-09")

    try:
        nightgate.main(["--state", str(sp)])
    except SystemExit as e:
        assert e.code == 0
