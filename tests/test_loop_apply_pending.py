"""Tests for loop.apply_pending_deploy — only restarts when a deploy is due."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import apply_pending_deploy as app
from loop import state as state_mod


def test_no_restart_when_no_pending(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": None})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is False
    assert calls == []


def test_restarts_and_clears_stamp_when_due(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": "2026-06-11T03:00:00+02:00"})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is True
    assert calls == ["restart"]
    # Stamp cleared so we don't restart again on the next timer fire.
    assert state_mod.load_state(sp)["pending_restart_at"] is None


def test_no_restart_when_pending_in_future(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": "2026-06-11T05:00:00+02:00"})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is False
    assert calls == []
