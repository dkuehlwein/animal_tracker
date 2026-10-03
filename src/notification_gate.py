"""
Notification gating: the single source of truth for whether a processed
burst is sent to Telegram, muted, or deferred.

`decide(ctx)` walks an ordered gate list and returns the FIRST match:

    HUMAN-GATE       status is HUMAN and suppress_human_alerts is on
    HUMAN-PROXIMITY  human_proximity_muted, for a review-class burst or an
                     IDENTIFIED burst with the generic unnamed-animal label
    BLUR             review-class, below the sharpness floor, and the frame
                     is bright enough (luma known and >= blur_mute_min_luma)
    BLANK-CONF       review-class and blank_confidence_muted
    REVIEW-SAMPLE    review-class and review_sampled_out
    (DEFER)          review-class and review_defer_seconds > 0
    (SEND)           everything else

Because the first match wins, a suppressed burst always produces exactly one
suppression log line, tagged with the gate that decided it.

The per-gate inputs (human_proximity_muted, blank_confidence_muted,
review_sampled_out, below_sharpness_floor) are computed earlier, in
`WildlifeSystem.process_detection`, and persisted to their own DB columns
with unchanged semantics — the nightly loop's metrics read those columns.
`decide()` only turns them into one routing decision. The two evaluators
that compute those flags from raw inputs (`evaluate_human_proximity`,
`evaluate_blank_confidence`) live here too, next to the store of recent
HUMAN-status timestamps they read (`RecentHumanEvents`).
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from data_models import is_human_detection, is_review_detection
from utils import is_blank_label

logger = logging.getLogger(__name__)


class Action:
    SEND = "send"
    MUTE = "mute"
    DEFER = "defer"


class Channel:
    MAIN = "main"
    REVIEW = "review"


@dataclass(frozen=True)
class Decision:
    action: str
    channel: str
    gate: Optional[str] = None  # log tag without brackets; None for SEND/DEFER
    reason: str = ""

    def log_line(self, detection_id) -> str:
        return (f"[{self.gate}] Suppressing notification for detection "
                f"{detection_id} ({self.reason})")


@dataclass(frozen=True)
class GateContext:
    """Everything the gates read. `config` is the PerformanceConfig (gate
    levers and the values quoted in the log reasons)."""
    status: Any
    config: Any
    unnamed_animal: bool = False
    human_proximity_muted: bool = False
    human_proximity_reason: Optional[str] = None
    below_sharpness_floor: bool = False
    sharpness_score: Optional[float] = None
    luma: Optional[float] = None
    blank_confidence_muted: bool = False
    top_species_raw: Optional[str] = None
    top_species_score: Optional[float] = None
    review_sampled_out: bool = False


def _fmt(value, spec: str) -> str:
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "n/a"


def _human_proximity_detail(reason: Optional[str], cfg) -> str:
    if reason == "density":
        return (f"reason=density: >= {cfg.human_density_count} human detections "
                f"in the last {cfg.human_density_window_seconds:.0f}s")
    if reason == "demoted-band window":
        widened = max(cfg.human_proximity_window_seconds, cfg.human_demoted_window_seconds)
        return (f"reason=demoted-band window: within {widened:.0f}s of last human "
                f"detection (person_confidence >= {cfg.human_demoted_person_floor})")
    return (f"reason=window: within {cfg.human_proximity_window_seconds:.0f}s "
            f"of last human detection")


def decide(ctx: GateContext) -> Decision:
    """Ordered gate list, first match wins. Pure: reads only `ctx`."""
    cfg = ctx.config
    review = is_review_detection(ctx.status)
    channel = Channel.REVIEW if review else Channel.MAIN

    def mute(gate: str, reason: str) -> Decision:
        return Decision(Action.MUTE, channel, gate, reason)

    if cfg.suppress_human_alerts and is_human_detection(ctx.status):
        return mute("HUMAN-GATE", "status=human")

    if ctx.human_proximity_muted and (review or ctx.unnamed_animal):
        return mute("HUMAN-PROXIMITY",
                    f"{_human_proximity_detail(ctx.human_proximity_reason, cfg)}, "
                    f"no animal found")

    if (review and ctx.below_sharpness_floor
            and ctx.luma is not None and ctx.luma >= cfg.blur_mute_min_luma):
        return mute("BLUR",
                    f"sharpness={_fmt(ctx.sharpness_score, '.1f')}, "
                    f"luma={_fmt(ctx.luma, '.1f')}, no animal found")

    if review and ctx.blank_confidence_muted:
        return mute("BLANK-CONF",
                    f"raw_top1={ctx.top_species_raw}, "
                    f"score={_fmt(ctx.top_species_score, '.3f')} >= "
                    f"threshold={_fmt(cfg.blank_confidence_mute_threshold, '.3f')}, "
                    f"no animal found")

    if review and ctx.review_sampled_out:
        return mute("REVIEW-SAMPLE",
                    f"sampled out, rate={_fmt(cfg.review_sample_rate, '.3f')}")

    if review and cfg.review_defer_seconds > 0:
        return Decision(Action.DEFER, channel)

    return Decision(Action.SEND, channel)


# ---------------------------------------------------------------------------
# Recent HUMAN-status events
# ---------------------------------------------------------------------------

class RecentHumanEvents:
    """Sorted store of recent HUMAN-status detection timestamps (capture time).

    One container serves every human-relative check: the backward window,
    demoted-band and density conditions of the Human-Proximity gate, and the
    forward cancel-on-human check of the deferred REVIEW send. Asking "is
    there ANY human in this interval" (rather than "is the latest one in
    it") is what makes the forward check correct when a later human
    arrives before the deferred task wakes (review bug HIGH-1).

    Pruning is purely a memory bound: every query filters by its own
    interval, so keeping extra old entries never changes a result.
    """

    def __init__(self, times: Optional[Iterable[datetime]] = None):
        self._times: List[datetime] = sorted(times or [])

    def seed(self, times: Iterable[datetime]) -> None:
        self._times = sorted(times)

    def add(self, t: datetime) -> None:
        bisect.insort(self._times, t)

    def prune(self, reference: datetime, horizon_seconds: float) -> None:
        cutoff = reference - timedelta(seconds=horizon_seconds)
        self._times = self._times[bisect.bisect_left(self._times, cutoff):]

    def latest(self) -> Optional[datetime]:
        return self._times[-1] if self._times else None

    def latest_in(self, start: datetime, end: datetime) -> Optional[datetime]:
        """Most recent timestamp in the closed interval [start, end]."""
        i = bisect.bisect_right(self._times, end)
        if i and self._times[i - 1] >= start:
            return self._times[i - 1]
        return None

    def first_after(self, start: datetime, end: datetime) -> Optional[datetime]:
        """Earliest timestamp in the half-open interval (start, end]."""
        i = bisect.bisect_right(self._times, start)
        if i < len(self._times) and self._times[i] <= end:
            return self._times[i]
        return None

    def count_in(self, start: datetime, end: datetime) -> int:
        """Number of timestamps in the closed interval [start, end]."""
        return bisect.bisect_right(self._times, end) - bisect.bisect_left(self._times, start)

    def __len__(self) -> int:
        return len(self._times)

    def __iter__(self) -> Iterator[datetime]:
        return iter(list(self._times))


# Slack on top of the longest window, so an entry needed by a deferred
# check is never pruned while species ID (~17s) of a later burst runs.
HUMAN_EVENTS_PRUNE_SLACK_SECONDS = 600.0


def human_events_horizon_seconds(cfg) -> float:
    """How far back the store must reach: the longest backward window plus
    the forward defer window plus slack."""
    backward = max(
        cfg.human_proximity_window_seconds,
        cfg.human_demoted_window_seconds,
        cfg.human_density_window_seconds,
    )
    return backward + cfg.review_defer_seconds + HUMAN_EVENTS_PRUNE_SLACK_SECONDS


def evaluate_human_proximity(burst_time: datetime, person_confidence: Optional[float],
                             events: RecentHumanEvents, cfg) -> Tuple[bool, Optional[str]]:
    """Human-Proximity gate conditions for one burst: (muted, reason).

    - window: a HUMAN detection in [burst_time - window, burst_time]; the
      window widens to max(window, human_demoted_window_seconds) when this
      burst's own person_confidence >= human_demoted_person_floor (reason
      "demoted-band window" only when the widening is what caused the mute).
    - density: >= human_density_count HUMAN detections in
      [burst_time - human_density_window_seconds, burst_time].
    A 0 lever disables its condition. The caller wraps this in a fail-open
    try/except; the demoted widening fails open to the base window here.
    """
    window = cfg.human_proximity_window_seconds

    demoted_active = False
    demoted_window = 0.0
    try:
        demoted_window = cfg.human_demoted_window_seconds
        demoted_active = bool(
            demoted_window > 0
            and person_confidence is not None
            and person_confidence >= cfg.human_demoted_person_floor
        )
    except Exception:
        demoted_active = False

    effective_window = max(window, demoted_window) if demoted_active else window

    window_muted = bool(
        effective_window > 0
        and events.latest_in(burst_time - timedelta(seconds=effective_window), burst_time)
        is not None
    )
    flat_window_hit = (
        events.latest_in(burst_time - timedelta(seconds=window), burst_time) is not None
    )

    density_threshold = cfg.human_density_count
    density_muted = bool(
        density_threshold > 0
        and events.count_in(
            burst_time - timedelta(seconds=cfg.human_density_window_seconds), burst_time
        ) >= density_threshold
    )

    if window_muted:
        return True, ("window" if flat_window_hit or not demoted_active
                      else "demoted-band window")
    if density_muted:
        return True, "density"
    return False, None


def evaluate_blank_confidence(status, top_species_raw: Optional[str],
                              top_species_score: Optional[float],
                              threshold: float) -> Optional[bool]:
    """Confident-Blank gate flag: None when the gate doesn't apply (not
    review-class, or threshold 0.0 = disabled), else whether the classifier's
    raw top-1 is the generic blank label at or above `threshold`."""
    if threshold <= 0.0 or not is_review_detection(status):
        return None
    return bool(
        top_species_raw
        and is_blank_label(top_species_raw)
        and top_species_score is not None
        and top_species_score >= threshold
    )
