"""Tests for src/loop/state.py — atomic state.json read/write."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import state as state_mod


def test_load_missing_returns_default(tmp_path):
    p = tmp_path / "state.json"
    loaded = state_mod.load_state(p)
    assert loaded == {}


def test_save_then_load_roundtrips(tmp_path):
    p = tmp_path / "state.json"
    data = {"watermark": 5, "deployed": {"MOTION_THRESHOLD": 2500}}
    state_mod.save_state(p, data)
    assert state_mod.load_state(p) == data


def test_save_is_atomic_no_temp_left_behind(tmp_path):
    p = tmp_path / "state.json"
    state_mod.save_state(p, {"a": 1})
    leftovers = [f.name for f in tmp_path.iterdir() if f.name != "state.json"]
    assert leftovers == []
