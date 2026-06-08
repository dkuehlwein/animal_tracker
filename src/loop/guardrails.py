"""Guardrails for the autonomous tuning loop (ADR-004 Phase 4).

Single source of truth for:
- BOUNDS: allowed ranges for every tunable param (consumed by config.py field
  validators at load-time AND by deploy.py before any write).
- Fast guards: capture-volume collapse / explosion vs a trailing baseline.
- FN-veto: reject an FP win that comes with an FN rise beyond CI noise; HOLD a
  change when FN is unmeasured and the change could plausibly raise FN.
- Feedback-starved freeze: stop tuning when no fresh human labels for N days.
"""

from __future__ import annotations

from typing import Optional

# Keys are env-var names so config.py validators and deploy.py share one map.
# (low, high) inclusive. Ranges are deliberately conservative — the loop tunes
# within these; out-of-range is rejected by the SYSTEM, not merely discouraged.
BOUNDS: dict[str, tuple[float, float]] = {
    "MOTION_THRESHOLD": (200, 8000),
    "MOTION_MIN_CONTOUR_AREA": (10, 2000),
    "MOTION_CONSECUTIVE_REQUIRED": (1, 6),
    "MOTION_MIN_COLOR_VARIANCE": (0.0, 2000.0),
    "SPECIES_UNKNOWN_SPECIES_THRESHOLD": (0.3, 0.95),
}

FEEDBACK_STARVED_DAYS = 3


def validate_param(key: str, value: float) -> None:
    """Raise ValueError if `key` is not tunable or `value` is out of bounds."""
    if key not in BOUNDS:
        raise ValueError(f"{key!r} is not a tunable parameter")
    low, high = BOUNDS[key]
    if not (low <= value <= high):
        raise ValueError(
            f"{key}={value} out of bounds [{low}, {high}]"
        )


# Multipliers off the trailing baseline that signal something broke, not tuned.
VOLUME_COLLAPSE_FRACTION = 0.1   # < 10% of baseline ⇒ camera/loop likely dead
VOLUME_EXPLOSION_FACTOR = 5.0    # > 5x baseline ⇒ runaway false positives


def check_volume(tonight: int, baseline: float) -> dict:
    """Detect capture-volume collapse (~0) or explosion vs trailing baseline.

    Returns {"rollback": bool, "reason": str}. A baseline <= 0 is treated as
    "no baseline yet" → never recommends rollback (avoids div-by-zero / false
    alarm on a fresh install).
    """
    if baseline <= 0:
        return {"rollback": False, "reason": "no baseline yet"}
    if tonight <= baseline * VOLUME_COLLAPSE_FRACTION:
        return {
            "rollback": True,
            "reason": f"volume collapse: {tonight} vs baseline {baseline:.1f}",
        }
    if tonight >= baseline * VOLUME_EXPLOSION_FACTOR:
        return {
            "rollback": True,
            "reason": f"volume explosion: {tonight} vs baseline {baseline:.1f}",
        }
    return {"rollback": False, "reason": "volume within normal range"}


def fn_veto(
    fp_before: float,
    fp_after: float,
    fn_before: Optional[float],
    fn_before_ci: Optional[tuple[float, float]],
    fn_after: Optional[float],
    fn_after_ci: Optional[tuple[float, float]],
    fn_risk: str = "low",
) -> dict:
    """Veto an FP win that worsens (or risks worsening) FN.

    Decision values:
      - "accept": FP improved and FN did not rise beyond CI noise.
      - "reject": FP improved but FN rose beyond CI noise (after-CI low > before-CI high).
      - "hold":   FN unmeasured and the change could plausibly raise FN (fn_risk != "low").

    A zero/None FN must NOT silently clear the veto — that is exactly the failure
    the spec warns about ("a zero would falsely clear the FN-veto").
    """
    fp_improved = fp_after < fp_before

    fn_unmeasured = fn_after is None or fn_after_ci is None or fn_before_ci is None
    if fn_unmeasured:
        if fn_risk != "low":
            return {
                "decision": "hold",
                "reason": "FN unmeasured and change could raise FN; holding",
            }
        return {
            "decision": "accept" if fp_improved else "reject",
            "reason": "FN unmeasured but change is low FN-risk",
        }

    # Both FN intervals known: a real rise = after's lower bound above before's upper.
    fn_rose_significantly = fn_after_ci[0] > fn_before_ci[1]
    if fp_improved and fn_rose_significantly:
        return {
            "decision": "reject",
            "reason": (
                f"FN rose beyond CI: after {fn_after_ci} > before {fn_before_ci}"
            ),
        }
    if fp_improved:
        return {"decision": "accept", "reason": "FP improved, FN stable within CI"}
    return {"decision": "reject", "reason": "no FP improvement"}


def is_feedback_starved(days_since_last_label: int) -> bool:
    """Freeze tuning if no fresh human labels for >= FEEDBACK_STARVED_DAYS days."""
    return days_since_last_label >= FEEDBACK_STARVED_DAYS
