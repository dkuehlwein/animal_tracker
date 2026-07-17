"""
Unit tests for DatabaseManager Phase-1 additions (ADR-004):
WAL, schema migration, richer detection logging, and the feedback table.
"""

import sqlite3
import sys
from datetime import datetime, timedelta
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
        background_drift=12.5,
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
    assert row["background_drift"] == pytest.approx(12.5)
    assert 0 <= row["hour_of_day"] <= 23  # derived from insert time


def test_log_detection_persists_observability_fields(tmp_path):
    """Task 1 (ADR-004 observability): sharpness_score, below_sharpness_floor,
    and person_confidence round-trip through log_detection so the nightly
    tuning loop can attribute metric shifts to blur/person signals."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_2.jpg",
        motion_area=1200,
        sharpness_score=15.2,
        below_sharpness_floor=False,
        person_confidence=0.25,
    )
    assert det_id is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["sharpness_score"] == pytest.approx(15.2)
    assert row["below_sharpness_floor"] == 0
    assert row["person_confidence"] == pytest.approx(0.25)


def test_log_detection_observability_fields_default_null(tmp_path):
    """Old call signature (no observability kwargs) still works; new cols are NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c2.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["sharpness_score"] is None
    assert row["below_sharpness_floor"] is None
    assert row["person_confidence"] is None


def test_log_detection_persists_top_species_guess(tmp_path):
    """Task 3 (ADR-004 observability): top_species_raw/top_species_score
    round-trip through log_detection so the caption and the nightly tuning
    loop can both see the classifier's raw top-1 prediction, distinct from
    the (possibly rolled-up) ensemble species_name."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_3.jpg",
        motion_area=1200,
        top_species_raw="def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird",
        top_species_score=0.34,
    )
    assert det_id is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["top_species_raw"] == "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird"
    assert row["top_species_score"] == pytest.approx(0.34)


def test_log_detection_top_species_guess_defaults_null(tmp_path):
    """Old call signature (no top-species kwargs) still works; new columns NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c3.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["top_species_raw"] is None
    assert row["top_species_score"] is None


def test_log_detection_persists_scene_gate_fields(tmp_path):
    """Task 2 (scene-unchanged gate): scene_similarity/scene_gate_muted
    round-trip through log_detection so the gate's decision is auditable."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_4.jpg",
        motion_area=1200,
        scene_similarity=0.97,
        scene_gate_muted=True,
    )
    assert det_id is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["scene_similarity"] == pytest.approx(0.97)
    assert row["scene_gate_muted"] == 1


def test_log_detection_scene_gate_fields_default_null(tmp_path):
    """Old call signature (no scene-gate kwargs) still works; new cols are NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c4.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["scene_similarity"] is None
    assert row["scene_gate_muted"] is None


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


# ---------------------------------------------------------------------------
# Timezone correctness: timestamps must be local wall-clock time, not UTC.
# SQLite's CURRENT_TIMESTAMP is always UTC; we must NOT rely on it.
# The system runs in CEST (UTC+2 in summer) and image filenames already use
# local time; DB timestamps must agree with them.
# ---------------------------------------------------------------------------

def test_log_detection_timestamp_is_local_time(tmp_path):
    """detections.timestamp must be stored as local wall-clock time.

    We record the local time just before and after the insert; the stored
    value must fall within that window.  On a UTC system the default
    CURRENT_TIMESTAMP would be ~2 hours behind, so this test would fail
    there — which is exactly the bug we're fixing.
    """
    db, db_path = _make_db(tmp_path)

    before = datetime.now().replace(microsecond=0)
    det_id = db.log_detection(image_path="capture_20260608_191715_frame1.jpg", motion_area=500)
    after = datetime.now()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT timestamp FROM detections WHERE id = ?", (det_id,)
        ).fetchone()

    stored = datetime.fromisoformat(row[0])
    assert before <= stored <= after, (
        f"Expected local time in [{before}, {after}] but got {stored}. "
        "Is the timestamp being written in UTC instead of local time?"
    )


def test_add_feedback_created_at_is_local_time(tmp_path):
    """detection_feedback.created_at must be stored as local wall-clock time.

    Same reasoning as test_log_detection_timestamp_is_local_time: the
    SQLite DEFAULT CURRENT_TIMESTAMP is UTC; we must write local time
    explicitly.
    """
    db, db_path = _make_db(tmp_path)

    det_id = db.log_detection(image_path="c.jpg", motion_area=10)

    before = datetime.now().replace(microsecond=0)
    fb_id = db.add_feedback(det_id, "animal")
    after = datetime.now()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT created_at FROM detection_feedback WHERE id = ?", (fb_id,)
        ).fetchone()

    stored = datetime.fromisoformat(row[0])
    assert before <= stored <= after, (
        f"Expected local time in [{before}, {after}] but got {stored}. "
        "Is created_at being written in UTC instead of local time?"
    )


# ---------------------------------------------------------------------------
# get_human_detections_older_than (Task 3: 48h privacy purge)
# ---------------------------------------------------------------------------

def _age_row(db_path, detection_id, hours_ago):
    """Back-date a detections.timestamp for purge-window testing."""
    ts = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE detections SET timestamp = ? WHERE id = ?", (ts, detection_id))
        conn.commit()
    return ts


def test_get_human_detections_older_than_returns_old_human_row(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_old_frame1.jpg", motion_area=10, detection_status="human"
    )
    ts = _age_row(db_path, det_id, hours_ago=49)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_detections_older_than(cutoff)

    assert rows == [(det_id, "capture_old_frame1.jpg", ts)]


def test_get_human_detections_older_than_excludes_recent_human_row(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_recent_frame1.jpg", motion_area=10, detection_status="human"
    )
    _age_row(db_path, det_id, hours_ago=1)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_detections_older_than(cutoff)

    assert rows == []


def test_get_human_detections_older_than_excludes_non_human_row(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_old_animal_frame1.jpg", motion_area=10,
        detection_status="identified",
    )
    _age_row(db_path, det_id, hours_ago=100)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_detections_older_than(cutoff)

    assert rows == []


# ---------------------------------------------------------------------------
# get_recent_review_detections (Task 2: scene-unchanged gate seed query)
# ---------------------------------------------------------------------------

def test_get_recent_review_detections_includes_no_animal_and_unclassifiable(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_no_animal = db.log_detection(
        image_path="capture_na.jpg", motion_area=10, detection_status="no_animal"
    )
    id_unclassifiable = db.log_detection(
        image_path="capture_unc.jpg", motion_area=10, detection_status="unclassifiable"
    )
    _age_row(db_path, id_no_animal, hours_ago=1)
    _age_row(db_path, id_unclassifiable, hours_ago=2)

    rows = db.get_recent_review_detections(limit=10, max_age_hours=24)

    paths = [r[0] for r in rows]
    assert "capture_na.jpg" in paths
    assert "capture_unc.jpg" in paths
    for _, ts in rows:
        assert isinstance(ts, datetime)


def test_get_recent_review_detections_excludes_human_and_identified(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_human = db.log_detection(
        image_path="capture_human.jpg", motion_area=10, detection_status="human"
    )
    id_identified = db.log_detection(
        image_path="capture_identified.jpg", motion_area=10, detection_status="identified"
    )
    _age_row(db_path, id_human, hours_ago=1)
    _age_row(db_path, id_identified, hours_ago=1)

    rows = db.get_recent_review_detections(limit=10, max_age_hours=24)

    assert rows == []


def test_get_recent_review_detections_excludes_outside_age_window(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_old = db.log_detection(
        image_path="capture_old.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, id_old, hours_ago=48)

    rows = db.get_recent_review_detections(limit=10, max_age_hours=24)

    assert rows == []


def test_get_recent_review_detections_ordered_desc_and_limited(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_oldest = db.log_detection(
        image_path="capture_oldest.jpg", motion_area=10, detection_status="no_animal"
    )
    id_middle = db.log_detection(
        image_path="capture_middle.jpg", motion_area=10, detection_status="no_animal"
    )
    id_newest = db.log_detection(
        image_path="capture_newest.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, id_oldest, hours_ago=3)
    _age_row(db_path, id_middle, hours_ago=2)
    _age_row(db_path, id_newest, hours_ago=1)

    rows = db.get_recent_review_detections(limit=2, max_age_hours=24)

    assert len(rows) == 2
    assert [r[0] for r in rows] == ["capture_newest.jpg", "capture_middle.jpg"]
