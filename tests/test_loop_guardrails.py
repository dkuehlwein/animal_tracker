"""Tests for src/loop/guardrails.py — bounds, fast guards, FN-veto, freeze."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from loop import guardrails


def test_bounds_dict_has_seeded_tunables():
    # Bounds are keyed by env-var name (same keys deploy.py validates against).
    assert "MOTION_THRESHOLD" in guardrails.BOUNDS
    low, high = guardrails.BOUNDS["MOTION_THRESHOLD"]
    assert low < high


def test_validate_param_accepts_in_range():
    guardrails.validate_param("MOTION_THRESHOLD", 2000)  # no raise


def test_validate_param_rejects_out_of_range():
    low, high = guardrails.BOUNDS["MOTION_THRESHOLD"]
    with pytest.raises(ValueError, match="out of bounds"):
        guardrails.validate_param("MOTION_THRESHOLD", high + 1)


def test_validate_param_unknown_key_rejected():
    with pytest.raises(ValueError, match="not a tunable"):
        guardrails.validate_param("NOT_A_REAL_KEY", 1)


def test_volume_collapse_flags_near_zero():
    # Baseline ~40/night, tonight ~0 → collapse → rollback recommended.
    verdict = guardrails.check_volume(tonight=0, baseline=40.0)
    assert verdict["rollback"] is True
    assert "collapse" in verdict["reason"]


def test_volume_explosion_flags_spike():
    verdict = guardrails.check_volume(tonight=400, baseline=40.0)
    assert verdict["rollback"] is True
    assert "explos" in verdict["reason"]


def test_volume_normal_ok():
    verdict = guardrails.check_volume(tonight=45, baseline=40.0)
    assert verdict["rollback"] is False


def test_fn_veto_accepts_fp_win_with_stable_fn():
    # FP improved; FN measured and CIs overlap (no significant FN rise) → accept.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=0.10, fn_before_ci=(0.05, 0.18),
        fn_after=0.11, fn_after_ci=(0.06, 0.19),
    )
    assert verdict["decision"] == "accept"


def test_fn_veto_rejects_fp_win_with_fn_rise_beyond_ci():
    # FN jumps and the after-CI lower bound clears the before-CI upper bound.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=0.10, fn_before_ci=(0.05, 0.15),
        fn_after=0.30, fn_after_ci=(0.22, 0.40),
    )
    assert verdict["decision"] == "reject"
    assert "FN" in verdict["reason"]


def test_fn_veto_holds_when_fn_unmeasured_and_change_risky():
    # FN unmeasured (None) + change could raise FN → HOLD, never guess deploy.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=None, fn_before_ci=None,
        fn_after=None, fn_after_ci=None,
        fn_risk="high",
    )
    assert verdict["decision"] == "hold"


def test_freeze_when_no_labels_for_three_days():
    assert guardrails.is_feedback_starved(days_since_last_label=3) is True
    assert guardrails.is_feedback_starved(days_since_last_label=4) is True


def test_not_frozen_with_recent_labels():
    assert guardrails.is_feedback_starved(days_since_last_label=2) is False
    assert guardrails.is_feedback_starved(days_since_last_label=0) is False
