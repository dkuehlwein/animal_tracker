"""Tests for loop.scene_watch — new-scene-regime dead-man's switch.

Detects a camera re-aim/bump from saved burst frames by comparing edge
structure (not intensity — sun/shadow/exposure changes intensity, a re-aim
changes structure) of recent frames against a baseline window 3-7 days back.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import scene_watch


# ---------------------------------------------------------------------------
# should_alert_scene_change() — pure function
# ---------------------------------------------------------------------------

def test_none_measurement_no_alert():
    assert scene_watch.should_alert_scene_change(
        None, "2026-09-06", None
    ) is False


def test_fraction_above_threshold_no_alert():
    measurement = {"match_fraction": 0.5, "n_recent": 12, "n_baseline": 10, "n_matched": 6, "median_best": 0.6}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", None
    ) is False


def test_fraction_at_threshold_no_prior_alert_alerts():
    measurement = {"match_fraction": scene_watch.MATCH_FRACTION_THRESHOLD, "n_recent": 12, "n_baseline": 10, "n_matched": 3, "median_best": 0.1}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", None
    ) is True


def test_fraction_below_threshold_no_prior_alert_alerts():
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", None
    ) is True


def test_prior_alert_two_days_ago_suppressed():
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", "2026-09-04"
    ) is False


def test_prior_alert_nine_days_ago_alerts_again():
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", "2026-08-28"
    ) is True


def test_prior_alert_exactly_cooldown_days_ago_alerts_again():
    """Exactly ALERT_COOLDOWN_DAYS ago is a full cooldown-days gap — not
    suppressed (suppression is for STRICTLY fewer than cooldown_days)."""
    current = date(2026, 9, 6)
    last = current - timedelta(days=scene_watch.ALERT_COOLDOWN_DAYS)
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, current.isoformat(), last.isoformat()
    ) is True


def test_unparseable_last_alert_loopday_not_suppressed():
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", "not-a-date"
    ) is True


def test_missing_last_alert_loopday_not_suppressed():
    measurement = {"match_fraction": 0.0, "n_recent": 12, "n_baseline": 10, "n_matched": 0, "median_best": 0.05}
    assert scene_watch.should_alert_scene_change(
        measurement, "2026-09-06", None
    ) is True


# ---------------------------------------------------------------------------
# Image helpers for synthetic test scenes
# ---------------------------------------------------------------------------

def _write_scene_a(path: Path, rng: np.random.Generator, brightness_shift: int = 0) -> None:
    """A scene with a strong vertical structural edge (e.g. a fence post)."""
    img = np.full((240, 320), 100, dtype=np.uint8)
    img[:, 150:170] = 200
    img[40:80, 40:120] = 60
    noise = rng.integers(-5, 6, size=img.shape)
    img = np.clip(img.astype(int) + noise + brightness_shift, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def _write_scene_b(path: Path, rng: np.random.Generator, brightness_shift: int = 0) -> None:
    """A structurally different scene (e.g. camera now points at open lawn)."""
    img = np.full((240, 320), 140, dtype=np.uint8)
    img[180:240, :] = 90
    img[20:60, 200:280] = 220
    img[100:140, 20:100] = 40
    noise = rng.integers(-5, 6, size=img.shape)
    img = np.clip(img.astype(int) + noise + brightness_shift, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# edge_signature()
# ---------------------------------------------------------------------------

def test_edge_signature_shape_and_normalization(tmp_path):
    rng = np.random.default_rng(1)
    p = tmp_path / "img.jpg"
    _write_scene_a(p, rng)

    sig = scene_watch.edge_signature(p)
    assert sig is not None
    assert sig.shape == scene_watch.COMPARE_SIZE
    assert abs(float(sig.mean())) < 1e-3
    assert abs(float(sig.std()) - 1.0) < 1e-3


def test_edge_signature_nonexistent_path_returns_none(tmp_path):
    assert scene_watch.edge_signature(tmp_path / "does_not_exist.jpg") is None


def test_edge_signature_undecodable_file_returns_none(tmp_path):
    p = tmp_path / "not_an_image.jpg"
    p.write_text("this is definitely not image bytes")
    assert scene_watch.edge_signature(p) is None


# ---------------------------------------------------------------------------
# edge_similarity()
# ---------------------------------------------------------------------------

def test_edge_similarity_identical_signature_near_one(tmp_path):
    rng = np.random.default_rng(2)
    p = tmp_path / "img.jpg"
    _write_scene_a(p, rng)
    sig = scene_watch.edge_signature(p)
    assert scene_watch.edge_similarity(sig, sig) == pytest.approx(1.0, abs=1e-6)


def test_edge_similarity_different_scenes_clearly_lower(tmp_path):
    rng = np.random.default_rng(3)
    pa = tmp_path / "a.jpg"
    pb = tmp_path / "b.jpg"
    _write_scene_a(pa, rng)
    _write_scene_b(pb, rng)
    sig_a = scene_watch.edge_signature(pa)
    sig_b = scene_watch.edge_signature(pb)

    same_scene_sim = scene_watch.edge_similarity(sig_a, sig_a)
    diff_scene_sim = scene_watch.edge_similarity(sig_a, sig_b)
    assert diff_scene_sim < same_scene_sim
    assert diff_scene_sim < scene_watch.EDGE_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# measure_scene_match()
# ---------------------------------------------------------------------------

def _ts_name(ts: datetime) -> str:
    return f"capture_{ts.strftime('%Y%m%d_%H%M%S')}_frame1.jpg"


def _populate(image_dir: Path, now: datetime, rng: np.random.Generator,
              n_recent: int, n_baseline: int, recent_writer=_write_scene_a,
              baseline_writer=_write_scene_a, corrupt_one_recent=False):
    image_dir.mkdir(parents=True, exist_ok=True)

    # Baseline window: 3..7 days back, spread across the window.
    for i in range(n_baseline):
        frac = i / max(n_baseline - 1, 1)
        days_back = 7 - frac * 4  # spans 7 down to 3 days back
        ts = now - timedelta(days=days_back, minutes=i)
        baseline_writer(image_dir / _ts_name(ts), rng, brightness_shift=int(rng.integers(-10, 10)))

    # Recent window: most recent n_recent frames, right before `now`.
    for i in range(n_recent):
        ts = now - timedelta(minutes=(n_recent - i) * 10)
        p = image_dir / _ts_name(ts)
        if corrupt_one_recent and i == 0:
            p.write_text("corrupt, not an image")
        else:
            recent_writer(p, rng, brightness_shift=int(rng.integers(-10, 10)))


def test_measure_scene_match_same_scene_full_match(tmp_path):
    rng = np.random.default_rng(10)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(tmp_path, now, rng, n_recent=scene_watch.RECENT_FRAMES, n_baseline=15)

    result = scene_watch.measure_scene_match(tmp_path, now)
    assert result is not None
    assert result["match_fraction"] == pytest.approx(1.0)
    assert result["n_recent"] == scene_watch.RECENT_FRAMES


def test_measure_scene_match_different_scene_zero_match(tmp_path):
    rng = np.random.default_rng(11)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(
        tmp_path, now, rng,
        n_recent=scene_watch.RECENT_FRAMES, n_baseline=15,
        recent_writer=_write_scene_b, baseline_writer=_write_scene_a,
    )

    result = scene_watch.measure_scene_match(tmp_path, now)
    assert result is not None
    assert result["match_fraction"] == pytest.approx(0.0)


def test_measure_scene_match_too_few_recent_returns_none(tmp_path):
    """Too few frames captured near `now` -> None, even with no baseline
    frames to (incorrectly) backfill the recent window from."""
    rng = np.random.default_rng(12)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(
        tmp_path, now, rng,
        n_recent=scene_watch.MIN_RECENT_FRAMES - 1, n_baseline=0,
    )

    assert scene_watch.measure_scene_match(tmp_path, now) is None


def test_measure_scene_match_too_few_baseline_returns_none(tmp_path):
    rng = np.random.default_rng(13)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(
        tmp_path, now, rng,
        n_recent=scene_watch.RECENT_FRAMES, n_baseline=scene_watch.MIN_BASELINE_FRAMES - 1,
    )

    assert scene_watch.measure_scene_match(tmp_path, now) is None


def test_measure_scene_match_skips_corrupt_recent_frame(tmp_path):
    rng = np.random.default_rng(14)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(
        tmp_path, now, rng,
        n_recent=scene_watch.RECENT_FRAMES, n_baseline=15,
        corrupt_one_recent=True,
    )

    result = scene_watch.measure_scene_match(tmp_path, now)
    assert result is not None
    # The corrupt frame is skipped, not counted as recent for the fraction.
    assert result["n_recent"] == scene_watch.RECENT_FRAMES - 1
    assert result["match_fraction"] == pytest.approx(1.0)


def test_measure_scene_match_ignores_non_matching_filenames(tmp_path):
    rng = np.random.default_rng(15)
    now = datetime(2026, 9, 6, 20, 0, 0)
    _populate(tmp_path, now, rng, n_recent=scene_watch.RECENT_FRAMES, n_baseline=15)

    # Stray files that must be ignored entirely.
    _write_scene_a(tmp_path / "squirrel_test.png", rng)
    _write_scene_a(tmp_path / "manual_amsel_20260426_084927.jpg", rng)

    result = scene_watch.measure_scene_match(tmp_path, now)
    assert result is not None
    assert result["n_recent"] == scene_watch.RECENT_FRAMES


def test_measure_scene_match_never_raises_on_bad_dir():
    # Nonexistent directory: glob returns nothing, should return None cleanly.
    assert scene_watch.measure_scene_match(Path("/nonexistent/dir/xyz"), datetime.now()) is None
