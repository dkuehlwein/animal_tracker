"""Tests for /pause and /rollback pure-logic helpers in telegram_feedback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import state as state_mod
import telegram_feedback as tf


def test_handle_pause_sets_paused_flag(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"paused": False})
    msg = tf.handle_pause(state_path=sp)
    assert state_mod.load_state(sp)["paused"] is True
    assert "paus" in msg.lower()


def test_handle_rollback_invokes_deploy_rollback(tmp_path):
    sp = tmp_path / "state.json"
    env = tmp_path / "deployed_config.env"
    state_mod.save_state(sp, {
        "deployed": {"MOTION_THRESHOLD": 2500},
        "best_known_good": {"MOTION_THRESHOLD": 2000},
        "history": [], "pending_restart_at": None,
    })
    msg = tf.handle_rollback(state_path=sp, env_path=env,
                             restart_at="2026-06-11T05:00:00+02:00")
    assert state_mod.load_state(sp)["deployed"]["MOTION_THRESHOLD"] == 2000
    assert "MOTION_THRESHOLD=2000" in env.read_text()
    assert "rollback" in msg.lower()
