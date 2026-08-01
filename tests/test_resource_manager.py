"""
Unit tests for StorageManager's human-burst privacy purge (Task 3, ADR-004).

Saved photos of HUMAN-status detections must be deleted after
`human_retention_hours` (default 48h), while the DB row is kept as a
metadata-only record. Non-human bursts are untouched by this sweep — they
remain governed only by the existing FIFO/max_images cleanup.
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append('src')

from config import Config
from database_manager import DatabaseManager
from resource_manager import StorageManager


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up environment variables after each test."""
    yield
    for key in [
        'PERFORMANCE_MAX_IMAGES', 'PERFORMANCE_HUMAN_RETENTION_HOURS', 'STORAGE_DATA_DIR',
        'PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS',
    ]:
        os.environ.pop(key, None)


def _make_config(tmp_path):
    os.environ['STORAGE_DATA_DIR'] = str(tmp_path)
    return Config.create_test_config()


def _age_detection(db, db_path, image_path, detection_status, hours_ago):
    """Log a detection and back-date its timestamp for purge-window testing."""
    det_id = db.log_detection(image_path=str(image_path), motion_area=10,
                               detection_status=detection_status)
    ts = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE detections SET timestamp = ? WHERE id = ?", (ts, det_id))
        conn.commit()
    return det_id


def _touch_burst(image_dir, base, frame_count=5, annotated_frames=()):
    """Create burst frame files (and optional annotated variants) on disk."""
    for i in range(1, frame_count + 1):
        (image_dir / f"{base}_frame{i}.jpg").touch()
    for i in annotated_frames:
        (image_dir / f"{base}_frame{i}_annotated.jpg").touch()


def test_purge_human_bursts_deletes_old_human_burst_keeps_db_row(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    base = "capture_20260101_000000"
    _touch_burst(config.storage.image_dir, base, frame_count=5, annotated_frames=(1,))
    det_id = _age_detection(
        db, db.db_path, config.storage.image_dir / f"{base}_frame1.jpg",
        detection_status="human", hours_ago=49,
    )

    deleted = storage.purge_human_bursts()

    remaining = list(config.storage.image_dir.glob(f"{base}*"))
    assert remaining == []
    assert deleted == 6  # 5 frames + 1 annotated

    # DB row (metadata-only) must survive.
    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute("SELECT id FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row is not None


def test_purge_human_bursts_leaves_recent_human_burst(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    base = "capture_20260102_000000"
    _touch_burst(config.storage.image_dir, base, frame_count=3)
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{base}_frame1.jpg",
        detection_status="human", hours_ago=1,
    )

    deleted = storage.purge_human_bursts()

    remaining = list(config.storage.image_dir.glob(f"{base}*"))
    assert len(remaining) == 3
    assert deleted == 0


def test_purge_human_bursts_leaves_old_non_human_burst(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    base = "capture_20260103_000000"
    _touch_burst(config.storage.image_dir, base, frame_count=3)
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{base}_frame1.jpg",
        detection_status="identified", hours_ago=100,
    )

    deleted = storage.purge_human_bursts()

    remaining = list(config.storage.image_dir.glob(f"{base}*"))
    assert len(remaining) == 3
    assert deleted == 0


def test_purge_human_bursts_missing_files_no_exception(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    base = "capture_20260104_000000"
    # No files ever created on disk for this burst (already gone).
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{base}_frame1.jpg",
        detection_status="human", hours_ago=72,
    )

    deleted = storage.purge_human_bursts()  # must not raise

    assert deleted == 0


def test_purge_human_bursts_noop_without_database(tmp_path):
    """StorageManager constructed without a database ref (legacy call sites)
    must not error; purge is simply a no-op."""
    config = _make_config(tmp_path)
    storage = StorageManager(config)

    assert storage.purge_human_bursts() == 0


# ---------------------------------------------------------------------------
# Human-adjacent review-burst purge (leading-edge fix, 2026-07-31): a
# no_animal/unclassifiable burst whose timestamp falls within
# human_retention_proximity_seconds of a HUMAN-status detection (before OR
# after) is purged the same way as HUMAN-status bursts themselves — the
# human-proximity mute gate only looks BACKWARD, so a burst just BEFORE the
# first HUMAN burst of a visit can leak a recognisable face and otherwise
# survive the full ~300-burst rotation.
# ---------------------------------------------------------------------------

def test_purge_human_bursts_also_purges_adjacent_review_burst(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    human_base = "capture_20260106_000000"
    review_base = "capture_20260106_000100"
    _touch_burst(config.storage.image_dir, human_base, frame_count=3)
    _touch_burst(config.storage.image_dir, review_base, frame_count=5, annotated_frames=(1,))

    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{human_base}_frame1.jpg",
        detection_status="human", hours_ago=49,
    )
    # Review burst 100s BEFORE the human burst — the leading-edge case this
    # fix targets (more hours_ago = further in the past).
    review_id = _age_detection(
        db, db.db_path, config.storage.image_dir / f"{review_base}_frame1.jpg",
        detection_status="no_animal", hours_ago=49 + 100 / 3600,
    )

    deleted = storage.purge_human_bursts()

    remaining_human = list(config.storage.image_dir.glob(f"{human_base}*"))
    remaining_review = list(config.storage.image_dir.glob(f"{review_base}*"))
    assert remaining_human == []
    assert remaining_review == []
    assert deleted == 3 + 6  # human burst (3 frames) + review burst (5 frames + 1 annotated)

    # DB row for the review burst (metadata-only) must survive.
    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute("SELECT id FROM detections WHERE id = ?", (review_id,)).fetchone()
    assert row is not None


def test_purge_human_bursts_leaves_non_adjacent_review_burst(tmp_path):
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    human_base = "capture_20260107_000000"
    review_base = "capture_20260107_001000"
    _touch_burst(config.storage.image_dir, human_base, frame_count=3)
    _touch_burst(config.storage.image_dir, review_base, frame_count=3)

    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{human_base}_frame1.jpg",
        detection_status="human", hours_ago=49,
    )
    # 500s gap — outside the default 240s proximity window.
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{review_base}_frame1.jpg",
        detection_status="no_animal", hours_ago=49 + 500 / 3600,
    )

    deleted = storage.purge_human_bursts()

    remaining_review = list(config.storage.image_dir.glob(f"{review_base}*"))
    assert len(remaining_review) == 3
    assert deleted == 3  # only the human burst


def test_purge_human_bursts_proximity_window_zero_disables_review_purge(tmp_path):
    """PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS=0 (the rollback lever)
    leaves an otherwise-adjacent, old-enough review burst untouched."""
    os.environ['PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS'] = '0'
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    human_base = "capture_20260108_000000"
    review_base = "capture_20260108_000100"
    _touch_burst(config.storage.image_dir, human_base, frame_count=3)
    _touch_burst(config.storage.image_dir, review_base, frame_count=3)

    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{human_base}_frame1.jpg",
        detection_status="human", hours_ago=49,
    )
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{review_base}_frame1.jpg",
        detection_status="no_animal", hours_ago=49 + 100 / 3600,
    )

    deleted = storage.purge_human_bursts()

    remaining_review = list(config.storage.image_dir.glob(f"{review_base}*"))
    assert len(remaining_review) == 3
    assert deleted == 3  # only the human burst


def test_cleanup_old_images_also_purges_human_bursts(tmp_path):
    """The existing cleanup pass (cleanup_old_images) must run the human
    purge too, so callers don't need to invoke it separately."""
    config = _make_config(tmp_path)
    db = DatabaseManager(config)
    storage = StorageManager(config, database=db)

    base = "capture_20260105_000000"
    _touch_burst(config.storage.image_dir, base, frame_count=4)
    _age_detection(
        db, db.db_path, config.storage.image_dir / f"{base}_frame1.jpg",
        detection_status="human", hours_ago=49,
    )

    storage.cleanup_old_images()

    remaining = list(config.storage.image_dir.glob(f"{base}*"))
    assert remaining == []
