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
    # Human/privacy gate person-confidence trigger (exp #14, 2026-08-30). The
    # gate consumed MegaDetector person boxes *below* MegaDetector's own 0.5
    # operating threshold, so sub-threshold noise on empty frames was routed to
    # DetectionStatus.HUMAN (32/32 adjudicated bursts at pc 0.17-0.47 contained
    # no person). Tunable so the loop can align the gate with the detector's
    # operating point. Floored at the shipped 0.3 default — lowering it only
    # makes the phantom-HUMAN class larger — and capped at 0.7 so the loop can
    # never gut the privacy gate.
    "SPECIES_HUMAN_DETECTION_CONFIDENCE": (0.3, 0.7),
    "PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD": (0.80, 1.0),
    # Boolean flag, not a range — present only so loop.deploy is allowed to
    # flip it (deploy rejects keys not in BOUNDS). No config field_validator
    # consumes this entry; booleans don't range-check. See Task 6 /
    # experiments/PROTOCOL.md "Scene-gate ownership".
    "PERFORMANCE_SCENE_GATE_ENABLED": (0, 1),
    "PERFORMANCE_BLUR_MUTE_MIN_LUMA": (0.0, 255.0),
    # REVIEW-channel sampling gate: fraction of surviving review-class
    # bursts actually sent to Telegram. 1.0 = send everything (rollback
    # lever); 0.0 = send nothing.
    "PERFORMANCE_REVIEW_SAMPLE_RATE": (0.0, 1.0),
    # Human-proximity mute gate: look-back window (seconds) after the most
    # recent HUMAN-status detection during which review-class bursts are
    # muted. 0.0 = disabled (rollback lever).
    "PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS": (0.0, 600.0),
    # Human-density condition (2026-07-28, exp #11 extension): OR-ed onto the
    # human-proximity gate above to catch leaks that fall OUTSIDE the
    # look-back window during a long human-occupied session (e.g. gardening;
    # measured gaps of 432s/732s past the last human burst). Mutes a
    # review-class burst when at least PERFORMANCE_HUMAN_DENSITY_COUNT
    # HUMAN-status detections occurred in the trailing
    # PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS. 0 count = disabled (rollback
    # lever).
    "PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS": (0.0, 7200.0),
    "PERFORMANCE_HUMAN_DENSITY_COUNT": (0, 100),
    # Demoted-band window (exp #27, 2026-09-15): widens the window condition
    # above — not a third independent condition — to
    # max(PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS,
    # PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS) when the burst's own
    # person_confidence >= PERFORMANCE_HUMAN_DEMOTED_PERSON_FLOOR. Burst 5305
    # (person_confidence 0.436) leaked past both existing conditions at 480s
    # since the last HUMAN burst with only 5 in the trailing 1800s; replaying
    # the corpus found the max person_confidence over all human-labelled
    # animal rows is 0.0789, a clean >4x separation from the floor. 0.0
    # PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS disables the widening
    # (rollback lever) and restores the flat window behaviour.
    "PERFORMANCE_HUMAN_DEMOTED_PERSON_FLOOR": (0.0, 1.0),
    "PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS": (0.0, 7200.0),
    # Burst image retention (Change 2, 2026-07-27): raised from the 100
    # default so the nightly tuning loop can still visually adjudicate its
    # own muted/sampled-out bursts on busy days.
    "PERFORMANCE_MAX_IMAGES": (50, 500),
    # Leading-edge fix (2026-07-31, burst 3909 leaked a face 81s before the
    # visit's first HUMAN burst): the human-proximity gate is
    # backward-looking only, so a deferred-send-with-cancel-on-human gate
    # delays a review-class Telegram send by this many seconds and cancels
    # it if a HUMAN-status detection lands within the window. 0 = disabled
    # (rollback lever) — reviews send immediately, as before this fix.
    "PERFORMANCE_REVIEW_DEFER_SECONDS": (0.0, 600.0),
    # Same fix, storage side: symmetric look-around window (seconds) used
    # by the retention purge to also delete photos of no_animal/
    # unclassifiable bursts that sit within this many seconds of a
    # HUMAN-status detection (before OR after), not just HUMAN-status
    # bursts themselves. 0 = disabled (rollback lever).
    "PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS": (0.0, 3600.0),
    # Burst human sweep (2026-09-09, exp #21): the human/privacy gate only
    # ever sees ONE frame per burst — the sharpest — and sharpness is
    # uncorrelated with whether a person is visible. Burst 5119 leaked a
    # child's face because frame5 (13.57) beat frame1 (13.41) by 1% and was
    # the only one of five frames that did NOT classify as human. When a
    # burst's sibling frames diverge from the selected frame by at least this
    # fraction of pixels, the selected frame no longer represents the burst,
    # so the most-divergent siblings are re-identified for a person.
    # 0.0 = disabled (rollback lever).
    "PERFORMANCE_HUMAN_SWEEP_DIVERGENCE_THRESHOLD": (0.0, 1.0),
    # Cap on how many sibling frames the sweep may re-identify per burst
    # (~10s each on the Pi). 0 = disabled (second rollback lever).
    "PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES": (0, 4),
    # Confident-Blank Mute Gate (exp #29, 2026-09-17): mutes a review-class
    # burst when the classifier's raw top-1 prediction is SpeciesNet's
    # fully-generic "blank" (empty frame) label at high confidence. Measured
    # over all 182 review-class rows with a blank raw top-1 and a recorded
    # top_species_score: the 5 rows ever human-labelled animal/
    # animal_wrong_id score 0.6431-0.8475 (ceiling 0.8475); 42 human-confirmed
    # false positives score median 0.9219, max 0.9825. Lower bound here is
    # deliberately max(animal-labelled blank score)=0.8475 + 0.02 = 0.87 (NOT
    # 0.0) so the autonomous loop can never deploy a threshold at or below the
    # measured animal ceiling — the config-level field_validator still allows
    # 0.0 so a human can disable the gate by hand (the rollback lever).
    "PERFORMANCE_BLANK_CONFIDENCE_MUTE_THRESHOLD": (0.87, 1.0),
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
