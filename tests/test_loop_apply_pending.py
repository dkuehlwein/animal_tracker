"""Tests for loop.apply_pending_deploy — only restarts when a deploy is due."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_default_restart_camera_uses_sudo_n(monkeypatch):
    """_restart_camera() must call sudo -n so it never blocks for a password."""
    captured: list[dict] = []

    def fake_run(cmd, **kwargs):
        captured.append({"cmd": list(cmd), "kwargs": kwargs})
        return MagicMock(returncode=0)

    monkeypatch.setattr("loop.apply_pending_deploy.subprocess.run", fake_run)
    app._restart_camera()

    assert len(captured) == 1, "subprocess.run called exactly once"
    call = captured[0]
    assert call["cmd"] == [
        "sudo", "-n", "systemctl", "restart", "wildlife-camera.service"
    ], f"unexpected argv: {call['cmd']}"
    assert call["kwargs"].get("check") is True, "check=True must be passed"
