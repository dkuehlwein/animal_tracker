"""Tests for src/loop/replay.py — STUB Layer-A seam (interface shape stable)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import replay


def test_replay_returns_skipped():
    result = replay.replay(candidate_config={"MOTION_THRESHOLD": 2500}, labeled_set=[])
    assert result.status == "skipped"
    assert result.reason == "not implemented"
    assert result.metrics is None


def test_replay_result_is_dataclass_with_stable_fields():
    result = replay.replay(candidate_config={}, labeled_set=[])
    # Interface contract the rest of the system relies on.
    assert hasattr(result, "status")
    assert hasattr(result, "reason")
    assert hasattr(result, "metrics")
