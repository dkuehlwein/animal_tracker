"""New-scene-regime dead-man's switch: detect a camera re-aim/bump.

On 2026-09-01/02 the camera was physically re-aimed onto a completely
different scene during a service outage. `systemctl is-active` stayed
`active` the whole time, so nightgate's existing camera-liveness check never
noticed — the tuning loop then ran three full nights of analysis against a
scene that no longer existed. Service liveness is not scene liveness.

This module compares the last few captured burst frames against a baseline
sample from 3-7 days ago. The comparison is done on **edge structure**, not
raw intensity: a re-aim changes what's in the frame (structure), while sun
position, shadows, and auto-exposure change how bright it is (intensity).
Comparing edge structure is what gives this detector its margin — do not
"simplify" it back to intensity, `src/scene_gate.py`'s intensity comparator
was measured on this same corpus and does NOT separate the classes here.

Calibrated on the real 4-week image corpus (do not re-derive): a wide gap
separates same-scene days (match_fraction 0.58-1.00, 11 days) from the three
post-re-aim days (match_fraction 0.00 on all three), with the 0.25 threshold
sitting comfortably in the middle of that gap.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Per-pair edge-NCC above which a recent frame "matches" a baseline frame.
EDGE_MATCH_THRESHOLD = 0.45
# Alert when the fraction of matching recent frames is <= this.
MATCH_FRACTION_THRESHOLD = 0.25
# How many of the most recent frames to check.
RECENT_FRAMES = 12
# Baseline window: frames from 3..7 days before `now`.
BASELINE_DAYS_BACK = (3, 7)
MIN_RECENT_FRAMES = 8
MIN_BASELINE_FRAMES = 8
MAX_BASELINE_SAMPLES = 30
# One alert per re-aim event, not one per day — a real re-aim keeps scoring
# low for days.
ALERT_COOLDOWN_DAYS = 7
COMPARE_SIZE = (128, 128)

_FRAME_TS_RE = re.compile(r"^capture_(\d{8}_\d{6})_frame1\.jpg$")


def edge_signature(image_path: Path | str) -> np.ndarray | None:
    """Return a normalized edge-magnitude signature for `image_path`.

    A re-aim changes scene *structure*; sun/shadow/exposure changes
    *intensity*. Comparing edge structure rather than intensity is what
    gives this detector its margin against day/night and weather shifts.

    Returns None (logged at WARNING) if the image can't be read. Never
    raises.
    """
    try:
        im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            log.warning("edge_signature: could not read image: %s", image_path)
            return None

        s = cv2.resize(im, COMPARE_SIZE).astype(np.float32)
        s = cv2.GaussianBlur(s, (5, 5), 0)

        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        return (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)
    except Exception:  # noqa: BLE001
        log.warning("edge_signature: failed for %s", image_path, exc_info=True)
        return None


def edge_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Normalized cross-correlation of two already-normalized signatures.

    Range is roughly [-1, 1]; 1.0 means identical edge structure.
    """
    return float(np.mean(sig_a * sig_b))


def _parse_frame_timestamp(path: Path) -> datetime | None:
    """Parse the `YYYYMMDD_HHMMSS` timestamp out of a `capture_*_frame1.jpg`
    filename. Returns None for any non-matching name (e.g. stray files like
    `squirrel_test.png` or `manual_amsel_20260426_084927.jpg`)."""
    match = _FRAME_TS_RE.match(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def measure_scene_match(image_dir: Path | str, now: datetime) -> dict | None:
    """Compare recent burst frames against a 3-7-day-old baseline sample.

    Returns None (never raises) when there isn't enough data to make a call
    — insufficient data must never produce an alert. Otherwise returns:
        {"match_fraction": float, "n_recent": int, "n_baseline": int,
         "n_matched": int, "median_best": float}
    """
    try:
        image_dir = Path(image_dir)
        frames: list[tuple[datetime, Path]] = []
        for p in image_dir.glob("capture_*_frame1.jpg"):
            ts = _parse_frame_timestamp(p)
            if ts is not None:
                frames.append((ts, p))
        frames.sort(key=lambda pair: pair[0])

        recent_all = [p for ts, p in frames if ts <= now]
        recent = recent_all[-RECENT_FRAMES:]

        low_days, high_days = BASELINE_DAYS_BACK
        baseline_lo = now - timedelta(days=high_days)
        baseline_hi = now - timedelta(days=low_days)
        baseline = [p for ts, p in frames if baseline_lo <= ts <= baseline_hi]
        if len(baseline) > MAX_BASELINE_SAMPLES:
            idx = np.linspace(0, len(baseline) - 1, MAX_BASELINE_SAMPLES)
            baseline = [baseline[int(round(i))] for i in idx]

        if len(recent) < MIN_RECENT_FRAMES or len(baseline) < MIN_BASELINE_FRAMES:
            return None

        baseline_sigs = []
        for p in baseline:
            sig = edge_signature(p)
            if sig is not None:
                baseline_sigs.append(sig)
        if not baseline_sigs:
            return None

        bests: list[float] = []
        for p in recent:
            sig = edge_signature(p)
            if sig is None:
                continue
            best = max(edge_similarity(sig, b_sig) for b_sig in baseline_sigs)
            bests.append(best)

        if not bests:
            return None

        n_matched = sum(1 for b in bests if b >= EDGE_MATCH_THRESHOLD)
        n_recent = len(bests)
        return {
            "match_fraction": n_matched / n_recent,
            "n_recent": n_recent,
            "n_baseline": len(baseline_sigs),
            "n_matched": n_matched,
            "median_best": float(np.median(bests)),
        }
    except Exception:  # noqa: BLE001
        log.warning("measure_scene_match: failed for %s", image_dir, exc_info=True)
        return None


def should_alert_scene_change(
    measurement: dict | None,
    current_loop_day: str,
    last_alert_loopday: str | None,
    cooldown_days: int = ALERT_COOLDOWN_DAYS,
) -> bool:
    """Pure decision function: True iff a scene-change alert should fire.

    - False if `measurement` is None (insufficient data).
    - False if the match fraction is above `MATCH_FRACTION_THRESHOLD`.
    - False if a prior alert was sent fewer than `cooldown_days` whole days
      ago (one alert per re-aim event, not one per day).
    - Otherwise True.
    """
    if measurement is None:
        return False
    if measurement["match_fraction"] > MATCH_FRACTION_THRESHOLD:
        return False

    if last_alert_loopday:
        try:
            last = date.fromisoformat(last_alert_loopday)
            current = date.fromisoformat(current_loop_day)
        except ValueError:
            last = None
            current = None
        if last is not None and current is not None:
            if (current - last).days < cooldown_days:
                return False

    return True
