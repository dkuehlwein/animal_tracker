"""Tests for loop.endtick — stamps completion into state.json.

Also tests the loop_day() helper (defined in loop.state) since both
nightgate and endtick depend on it and must agree on the mapping.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import state as state_mod
from loop import endtick


# ---------------------------------------------------------------------------
# loop_day() — maps the 18:00→06:00 overnight window to ONE stable date
# ---------------------------------------------------------------------------

def _local_dt(year, month, day, hour, minute=0, tz_offset_h=2):
    """Build a timezone-aware datetime in a fixed UTC+tz_offset_h zone."""
    tz = timezone(timedelta(hours=tz_offset_h))
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def test_loop_day_23h_maps_to_current_date():
    """23:00 maps to the calendar date of that day (23h - 12h = 11h, same date)."""
    # e.g. 2026-06-08 23:00 local → loop_day = 2026-06-08
    dt = _local_dt(2026, 6, 8, 23)
    assert state_mod.loop_day(dt) == "2026-06-08"


def test_loop_day_02h_maps_to_prior_date():
    """02:00 on June 9 maps to June 8 (02h - 12h wraps back one day)."""
    dt = _local_dt(2026, 6, 9, 2)
    assert state_mod.loop_day(dt) == "2026-06-08"


def test_loop_day_06h_maps_to_prior_date():
    """06:00 on June 9 maps to June 8 (06h - 12h wraps back one day)."""
    dt = _local_dt(2026, 6, 9, 6)
    assert state_mod.loop_day(dt) == "2026-06-08"


def test_loop_day_18h_maps_to_current_date():
    """18:00 on June 8 maps to June 8 (18h - 12h = 06h, same date)."""
    dt = _local_dt(2026, 6, 8, 18)
    assert state_mod.loop_day(dt) == "2026-06-08"


def test_loop_day_23h_and_02h_same_loop_day():
    """23:00 Jun 8 and 02:00 Jun 9 should map to the SAME loop-day."""
    dt_23 = _local_dt(2026, 6, 8, 23)
    dt_02 = _local_dt(2026, 6, 9, 2)
    assert state_mod.loop_day(dt_23) == state_mod.loop_day(dt_02)


def test_loop_day_06h_and_next_18h_differ():
    """06:00 Jun 9 (end of Jun 8 loop-day) vs 18:00 Jun 9 (start of Jun 9 loop-day) differ."""
    dt_06 = _local_dt(2026, 6, 9, 6)
    dt_18 = _local_dt(2026, 6, 9, 18)
    assert state_mod.loop_day(dt_06) != state_mod.loop_day(dt_18)
    assert state_mod.loop_day(dt_06) == "2026-06-08"
    assert state_mod.loop_day(dt_18) == "2026-06-09"


def test_loop_day_midnight_boundary():
    """Explicit midnight boundary: 00:00 Jun 9 maps to Jun 8 loop-day."""
    dt_midnight = _local_dt(2026, 6, 9, 0)
    assert state_mod.loop_day(dt_midnight) == "2026-06-08"


def test_loop_day_no_arg_returns_string():
    """loop_day() with no argument uses now and returns a YYYY-MM-DD string."""
    result = state_mod.loop_day()
    assert isinstance(result, str)
    # Basic format check: YYYY-MM-DD
    parts = result.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4
    assert len(parts[1]) == 2
    assert len(parts[2]) == 2


# ---------------------------------------------------------------------------
# endtick.stamp() — sets last_tick_completed_day and last_heartbeat_loopday
# ---------------------------------------------------------------------------

def test_endtick_stamps_completed_day(tmp_path):
    """stamp() sets last_tick_completed_day to loop_day() in state.json."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"watermark": 0})

    dt = _local_dt(2026, 6, 8, 23)
    endtick.stamp(str(sp), now=dt)

    st = state_mod.load_state(sp)
    assert st["last_tick_completed_day"] == "2026-06-08"


def test_endtick_stamps_heartbeat_loopday(tmp_path):
    """stamp() also sets last_heartbeat_loopday to loop_day() in state.json."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"watermark": 0})

    dt = _local_dt(2026, 6, 8, 23)
    endtick.stamp(str(sp), now=dt)

    st = state_mod.load_state(sp)
    assert st["last_heartbeat_loopday"] == "2026-06-08"


def test_endtick_is_idempotent(tmp_path):
    """Calling stamp() twice for the same loop-day does not change state."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"watermark": 5})

    dt = _local_dt(2026, 6, 8, 23)
    endtick.stamp(str(sp), now=dt)
    st_first = state_mod.load_state(sp)
    endtick.stamp(str(sp), now=dt)
    st_second = state_mod.load_state(sp)

    assert st_first == st_second


def test_endtick_preserves_existing_state_keys(tmp_path):
    """stamp() does not destroy other keys in state.json."""
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"watermark": 7, "paused": True, "last_metrics": {"date": "2026-06-07"}})

    dt = _local_dt(2026, 6, 8, 23)
    endtick.stamp(str(sp), now=dt)

    st = state_mod.load_state(sp)
    assert st["watermark"] == 7
    assert st["paused"] is True
    assert st["last_metrics"]["date"] == "2026-06-07"


def test_endtick_main_works(tmp_path):
    """endtick.main() CLI path stamps state and exits 0."""
    import sys as _sys
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"watermark": 0})

    # main() reads --state from argv
    saved_argv = _sys.argv
    try:
        _sys.argv = ["loop.endtick", "--state", str(sp)]
        endtick.main()
    except SystemExit as e:
        assert e.code == 0 or e.code is None
    finally:
        _sys.argv = saved_argv

    st = state_mod.load_state(sp)
    assert "last_tick_completed_day" in st
    assert "last_heartbeat_loopday" in st
