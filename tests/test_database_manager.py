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


def test_log_detection_review_sampled_out_default_null(tmp_path):
    """Old call signature (no review_sampled_out kwarg) still works; new
    column is NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c5.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["review_sampled_out"] is None


def test_update_review_sampled_out_true_roundtrip(tmp_path):
    """REVIEW-sampling gate: update_review_sampled_out(True) persists as 1,
    matching the follow-up-UPDATE pattern (detection_id only exists after
    the initial INSERT returns, see wildlife_system.process_detection)."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c6.jpg", motion_area=10)
    db.update_review_sampled_out(det_id, True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["review_sampled_out"] == 1


def test_update_review_sampled_out_false_roundtrip(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c7.jpg", motion_area=10)
    db.update_review_sampled_out(det_id, False)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["review_sampled_out"] == 0


def test_update_human_proximity_muted_true_roundtrip(tmp_path):
    """Leading-edge fix (2026-07-31): the deferred REVIEW-send gate persists
    a cancel-on-human decision via this follow-up UPDATE, modeled exactly on
    update_review_sampled_out above."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c8.jpg", motion_area=10)
    db.update_human_proximity_muted(det_id, True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["human_proximity_muted"] == 1


def test_update_human_proximity_muted_false_roundtrip(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c9.jpg", motion_area=10, human_proximity_muted=True)
    db.update_human_proximity_muted(det_id, False)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["human_proximity_muted"] == 0


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
# get_human_adjacent_review_detections (leading-edge fix, 2026-07-31): a
# symmetric look-around companion to get_human_detections_older_than, used
# by the Task 3 privacy purge to also catch review-class (no_animal/
# unclassifiable) bursts that really contain a person but were
# misclassified — the human-proximity mute gate only looks BACKWARD from a
# HUMAN burst, so a burst just BEFORE the first HUMAN burst of a visit can
# leak a recognisable face and then survive the full image retention
# window. "Adjacent" is measured in seconds via SQLite's strftime('%s', ...)
# differencing, not the lexicographic string cutoff comparison used
# elsewhere — nearby-in-time isn't a lexicographic property.
# ---------------------------------------------------------------------------

def test_get_human_adjacent_review_detections_includes_row_before_human(tmp_path):
    """The leading-edge case this fix targets: a review-class row shortly
    BEFORE a HUMAN-status row is adjacent and included."""
    db, db_path = _make_db(tmp_path)
    review_id = db.log_detection(
        image_path="capture_review1_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    human_id = db.log_detection(
        image_path="capture_human1.jpg", motion_area=10, detection_status="human"
    )
    # Review row 100s BEFORE the human row (more hours_ago = further in the past).
    review_ts = _age_row(db_path, review_id, hours_ago=49 + 100 / 3600)
    _age_row(db_path, human_id, hours_ago=49)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)

    assert rows == [(review_id, "capture_review1_frame1.jpg", review_ts)]


def test_get_human_adjacent_review_detections_includes_row_after_human(tmp_path):
    """Symmetry check: a review-class row shortly AFTER a HUMAN-status row
    (the direction the existing backward-looking proximity gate already
    covers at write-time) is also adjacent and included by this purge-time
    query."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human2.jpg", motion_area=10, detection_status="human"
    )
    review_id = db.log_detection(
        image_path="capture_review2_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, human_id, hours_ago=49 + 100 / 3600)
    review_ts = _age_row(db_path, review_id, hours_ago=49)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)

    assert rows == [(review_id, "capture_review2_frame1.jpg", review_ts)]


def test_get_human_adjacent_review_detections_excludes_non_adjacent_row(tmp_path):
    """A review-class row well outside the look-around window (500s, vs a
    240s window) is excluded."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human3.jpg", motion_area=10, detection_status="human"
    )
    review_id = db.log_detection(
        image_path="capture_review3_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, human_id, hours_ago=49)
    _age_row(db_path, review_id, hours_ago=49 + 500 / 3600)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)

    assert rows == []


def test_get_human_adjacent_review_detections_excludes_rows_newer_than_cutoff(tmp_path):
    """An adjacent pair that's too recent (inside the cutoff, e.g. still
    within human_retention_hours) is not yet purge-eligible."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human4.jpg", motion_area=10, detection_status="human"
    )
    review_id = db.log_detection(
        image_path="capture_review4_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, human_id, hours_ago=1)
    _age_row(db_path, review_id, hours_ago=1 + 50 / 3600)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)

    assert rows == []


def test_get_human_adjacent_review_detections_zero_window_returns_empty(tmp_path):
    """window_seconds<=0 disables this query entirely (the rollback lever),
    even for an otherwise-perfectly-adjacent, old-enough pair."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human5.jpg", motion_area=10, detection_status="human"
    )
    review_id = db.log_detection(
        image_path="capture_review5_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    _age_row(db_path, human_id, hours_ago=49)
    _age_row(db_path, review_id, hours_ago=49)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=0)

    assert rows == []


def test_get_human_adjacent_review_detections_skips_malformed_human_timestamp(tmp_path):
    """Perf rewrite (2026-08-01): the adjacency match now happens in Python
    (bisect over parsed HUMAN timestamps), not a SQL strftime EXISTS
    subquery — a malformed HUMAN-status timestamp must be skipped during
    parsing, never raise, and never spuriously match anything."""
    db, db_path = _make_db(tmp_path)
    bad_human_id = db.log_detection(
        image_path="capture_human_bad.jpg", motion_area=10, detection_status="human"
    )
    review_id = db.log_detection(
        image_path="capture_review_bad1_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE detections SET timestamp = ? WHERE id = ?",
            ("not-a-timestamp", bad_human_id),
        )
        conn.commit()
    _age_row(db_path, review_id, hours_ago=49)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)  # must not raise

    assert rows == []


def test_get_human_adjacent_review_detections_skips_malformed_review_timestamp(tmp_path):
    """A malformed review-row timestamp that still slips past the SQL
    cutoff-range query (invalid but lexicographically 'old'-looking) must
    be skipped during Python-side parsing, not raise — and a separate,
    well-formed adjacent pair must still be matched correctly alongside
    it."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human_ok.jpg", motion_area=10, detection_status="human"
    )
    good_review_id = db.log_detection(
        image_path="capture_review_ok_frame1.jpg", motion_area=10, detection_status="no_animal"
    )
    bad_review_id = db.log_detection(
        image_path="capture_review_bad2_frame1.jpg", motion_area=10, detection_status="no_animal"
    )

    _age_row(db_path, human_id, hours_ago=49)
    good_ts = _age_row(db_path, good_review_id, hours_ago=49 + 100 / 3600)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE detections SET timestamp = ? WHERE id = ?",
            ("2020-13-45 99:99:99", bad_review_id),
        )
        conn.commit()

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)  # must not raise

    assert rows == [(good_review_id, "capture_review_ok_frame1.jpg", good_ts)]


def test_get_human_adjacent_review_detections_excludes_identified_row(tmp_path):
    """Defense-in-depth: an IDENTIFIED (non-review-class) row adjacent to a
    HUMAN row is never returned — only no_animal/unclassifiable rows are
    purge-eligible via this path."""
    db, db_path = _make_db(tmp_path)
    human_id = db.log_detection(
        image_path="capture_human6.jpg", motion_area=10, detection_status="human"
    )
    animal_id = db.log_detection(
        image_path="capture_animal6.jpg", motion_area=10, detection_status="identified"
    )
    _age_row(db_path, human_id, hours_ago=49)
    _age_row(db_path, animal_id, hours_ago=49 + 10 / 3600)

    cutoff = datetime.now() - timedelta(hours=48)
    rows = db.get_human_adjacent_review_detections(cutoff, window_seconds=240)

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


# ---------------------------------------------------------------------------
# human_proximity_muted column + get_last_human_detection_time
# (Human-Proximity Mute Gate, 2026-07-27)
# ---------------------------------------------------------------------------

def test_log_detection_persists_human_proximity_muted_true(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_hp1.jpg",
        motion_area=1200,
        human_proximity_muted=True,
    )
    assert det_id is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["human_proximity_muted"] == 1


def test_log_detection_persists_human_proximity_muted_false(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="capture_hp2.jpg",
        motion_area=1200,
        human_proximity_muted=False,
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()

    assert row["human_proximity_muted"] == 0


def test_log_detection_human_proximity_muted_default_null(tmp_path):
    """Old call signature (no human_proximity_muted kwarg) still works; new
    column is NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="capture_hp3.jpg", motion_area=10)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
    assert row["human_proximity_muted"] is None


def test_get_last_human_detection_time_returns_most_recent_human_row(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_older = db.log_detection(
        image_path="capture_older_human.jpg", motion_area=10, detection_status="human"
    )
    id_newer = db.log_detection(
        image_path="capture_newer_human.jpg", motion_area=10, detection_status="human"
    )
    _age_row(db_path, id_older, hours_ago=2)
    newer_ts = _age_row(db_path, id_newer, hours_ago=1)

    result = db.get_last_human_detection_time()

    assert result == datetime.strptime(newer_ts, "%Y-%m-%d %H:%M:%S")


def test_get_last_human_detection_time_ignores_non_human_rows(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_animal = db.log_detection(
        image_path="capture_animal.jpg", motion_area=10, detection_status="identified"
    )
    _age_row(db_path, id_animal, hours_ago=1)

    result = db.get_last_human_detection_time()

    assert result is None


def test_get_last_human_detection_time_no_rows_returns_none(tmp_path):
    db, _ = _make_db(tmp_path)
    assert db.get_last_human_detection_time() is None


# ---------------------------------------------------------------------------
# get_recent_human_detection_times (human-density condition, exp #11
# extension, 2026-07-28)
# ---------------------------------------------------------------------------

def test_get_recent_human_detection_times_includes_rows_since_cutoff(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_a = db.log_detection(
        image_path="capture_h1.jpg", motion_area=10, detection_status="human"
    )
    id_b = db.log_detection(
        image_path="capture_h2.jpg", motion_area=10, detection_status="human"
    )
    ts_a = _age_row(db_path, id_a, hours_ago=1)
    ts_b = _age_row(db_path, id_b, hours_ago=0.5)

    since = datetime.now() - timedelta(hours=2)
    result = db.get_recent_human_detection_times(since)

    assert result == [
        datetime.strptime(ts_a, "%Y-%m-%d %H:%M:%S"),
        datetime.strptime(ts_b, "%Y-%m-%d %H:%M:%S"),
    ]


def test_get_recent_human_detection_times_excludes_rows_before_cutoff(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_old = db.log_detection(
        image_path="capture_h_old.jpg", motion_area=10, detection_status="human"
    )
    _age_row(db_path, id_old, hours_ago=3)

    since = datetime.now() - timedelta(hours=1)
    result = db.get_recent_human_detection_times(since)

    assert result == []


def test_get_recent_human_detection_times_excludes_non_human_rows(tmp_path):
    db, db_path = _make_db(tmp_path)
    id_animal = db.log_detection(
        image_path="capture_animal2.jpg", motion_area=10, detection_status="identified"
    )
    _age_row(db_path, id_animal, hours_ago=0.1)

    since = datetime.now() - timedelta(hours=1)
    result = db.get_recent_human_detection_times(since)

    assert result == []


def test_get_recent_human_detection_times_no_rows_returns_empty_list(tmp_path):
    db, _ = _make_db(tmp_path)
    assert db.get_recent_human_detection_times(datetime.now() - timedelta(hours=1)) == []
