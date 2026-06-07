"""
Unit tests for DatabaseManager Phase-1 additions (ADR-004):
WAL, schema migration, richer detection logging, and the feedback table.
"""

import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.append('src')

from database_manager import DatabaseManager
from exceptions import DatabaseOperationError


def _make_db(tmp_path):
    """DatabaseManager only reads config.storage.database_path, so a tiny stub
    keeps these tests free of .env / pydantic setup."""
    db_path = tmp_path / "detections.db"
    config = SimpleNamespace(storage=SimpleNamespace(database_path=str(db_path)))
    return DatabaseManager(config), str(db_path)


def _create_old_schema(db_path):
    """Write a pre-Phase-1 detections table (no richer columns)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                motion_area INTEGER,
                species_name TEXT DEFAULT 'Unknown species',
                confidence_score REAL DEFAULT 0.0,
                processing_time REAL,
                api_success BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.execute(
            "INSERT INTO detections (image_path, motion_area) VALUES (?, ?)",
            ("old_capture.jpg", 1234),
        )
        conn.commit()


def test_wal_enabled(tmp_path):
    db, db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_migration_adds_columns_to_existing_db(tmp_path):
    db_path = tmp_path / "detections.db"
    _create_old_schema(str(db_path))

    # Opening through DatabaseManager should migrate in place, non-destructively.
    config = SimpleNamespace(storage=SimpleNamespace(database_path=str(db_path)))
    DatabaseManager(config)

    with sqlite3.connect(str(db_path)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
        # Old row survives migration
        old = conn.execute("SELECT image_path, motion_area FROM detections").fetchone()

    for expected in DatabaseManager._DETECTION_EXTRA_COLUMNS:
        assert expected in cols, f"migration missing column {expected}"
    assert old == ("old_capture.jpg", 1234)


def test_migration_is_idempotent(tmp_path):
    db, db_path = _make_db(tmp_path)
    # Re-init over an already-migrated DB must not raise (no duplicate columns).
    config = SimpleNamespace(storage=SimpleNamespace(database_path=db_path))
    DatabaseManager(config)


def test_log_detection_persists_richer_fields(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_1.jpg",
        motion_area=5000,
        species_name="Fox",
        confidence_score=0.9,
        animals_detected=True,
        detection_count=2,
        max_detection_confidence=0.83,
        contour_count=4,
        largest_contour_area=3200,
        foreground_pixel_count=4100,
        gate_would_suppress=False,
        frame_stability=12.5,
    )
    assert det_id is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["animals_detected"] == 1
    assert row["detection_count"] == 2
    assert row["max_detection_confidence"] == pytest.approx(0.83)
    assert row["contour_count"] == 4
    assert row["largest_contour_area"] == 3200
    assert row["foreground_pixel_count"] == 4100
    assert row["gate_would_suppress"] == 0
    assert row["frame_stability"] == pytest.approx(12.5)
    assert 0 <= row["hour_of_day"] <= 23  # derived from insert time


def test_log_detection_backward_compatible(tmp_path):
    """Old call signature (no richer kwargs) still works; new cols are NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["animals_detected"] is None
    assert row["detection_count"] is None


def test_add_and_get_feedback_roundtrip(tmp_path):
    db, _ = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    fb_id = db.add_feedback(det_id, "false_positive")
    assert fb_id is not None

    rows = db.get_feedback(det_id)
    assert len(rows) == 1
    assert rows[0][1] == det_id
    assert rows[0][2] == "false_positive"
    assert rows[0][3] == "human"


def test_feedback_is_append_only(tmp_path):
    db, _ = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    db.add_feedback(det_id, "false_positive")
    db.add_feedback(det_id, "animal")  # a re-tap appends, never overwrites

    rows = db.get_feedback(det_id)
    assert [r[2] for r in rows] == ["false_positive", "animal"]


def test_invalid_feedback_label_rejected(tmp_path):
    db, _ = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    with pytest.raises(DatabaseOperationError):
        db.add_feedback(det_id, "totally_bogus")
    assert db.get_feedback(det_id) == []
