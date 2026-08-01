import bisect
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from config import Config
from data_models import DetectionRecord
from exceptions import DatabaseError, DatabaseConnectionError, DatabaseOperationError

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        # Ensure database path is absolute to avoid path resolution issues
        self.db_path = str(Path(config.storage.database_path).resolve())
        self.init_database()
    
    # Columns added to `detections` after the original schema (ADR-004 Phase 1).
    # Kept here so init/migration share one source of truth.
    _DETECTION_EXTRA_COLUMNS = {
        "animals_detected": "BOOLEAN",
        "detection_count": "INTEGER",
        "max_detection_confidence": "REAL",
        "contour_count": "INTEGER",
        "largest_contour_area": "INTEGER",
        "foreground_pixel_count": "INTEGER",
        "hour_of_day": "INTEGER",
        "gate_would_suppress": "BOOLEAN",
        "background_drift": "REAL",
        "detection_status": "TEXT",
        # Task 1 (ADR-004 observability): already-computed values that were
        # never persisted, so the nightly tuning loop can attribute metric
        # shifts to blur/person signals instead of just guessing.
        "sharpness_score": "REAL",
        "below_sharpness_floor": "BOOLEAN",
        "person_confidence": "REAL",
        # Task 3 (ADR-004 observability): the classifier's raw top-1
        # prediction (before geofence/rollup), distinct from the (possibly
        # rolled-up) ensemble species_name — lets the nightly loop and the
        # notification caption both see the more specific guess.
        "top_species_raw": "TEXT",
        "top_species_score": "REAL",
        # Task 2 (scene-unchanged gate): the frame comparator's similarity
        # score against the rolling empty-scene reference set, and whether
        # that score crossed the mute threshold — persisted so the gate's
        # decision is auditable and the nightly tuning loop can attribute
        # muted volume to it.
        "scene_similarity": "REAL",
        "scene_gate_muted": "BOOLEAN",
        # REVIEW-channel sampling gate: True when a review-class burst was
        # sampled OUT of Telegram notification, False when review-class and
        # sampled in, NULL when the status isn't review-class (or the gate
        # never evaluated it). Written via a follow-up UPDATE, not the
        # initial INSERT — see update_review_sampled_out below.
        "review_sampled_out": "BOOLEAN",
        # Human-proximity mute gate: True when a review-class burst landed
        # within human_proximity_window_seconds of the most recent
        # HUMAN-status detection, False when review-class and outside the
        # window, NULL when the status isn't review-class (or the gate never
        # evaluated it). Same NULL-for-non-review-class convention as
        # scene_gate_muted; unlike review_sampled_out this is written on the
        # initial INSERT (no detection_id dependency).
        "human_proximity_muted": "BOOLEAN",
    }

    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            # Ensure data directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # WAL lets the feedback sidecar write while the main process
                # writes detections (two writers, one DB). Set once; persists.
                cursor.execute('PRAGMA journal_mode=WAL')

                # Create detections table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
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

                # Add Phase-1 richer-logging columns to pre-existing databases.
                self._migrate_detection_columns(cursor)

                # Human/machine ground-truth labels keyed on detections.id.
                # Append-only by convention (never UPDATE/DELETE) — the
                # anti-self-poisoning rule from ADR-004.
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_id INTEGER NOT NULL,
                        label TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'human',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (detection_id) REFERENCES detections(id)
                    )
                ''')

                # Create species table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS species (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        first_detected DATETIME,
                        detection_count INTEGER DEFAULT 1
                    )
                ''')

                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                    ON detections(timestamp)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_detections_species
                    ON detections(species_name)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_species_name
                    ON species(name)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_detection
                    ON detection_feedback(detection_id)
                ''')

                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"Failed to initialize database: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error initializing database: {e}") from e

    def _migrate_detection_columns(self, cursor):
        """Idempotently add Phase-1 columns to an existing detections table.

        SQLite ALTER TABLE ADD COLUMN is cheap and the columns are nullable, so
        old rows remain valid. Guarded by PRAGMA so re-runs are no-ops.
        """
        cursor.execute("PRAGMA table_info(detections)")
        existing = {row[1] for row in cursor.fetchall()}
        for name, col_type in self._DETECTION_EXTRA_COLUMNS.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE detections ADD COLUMN {name} {col_type}")
                logger.info(f"Migrated detections table: added column '{name}'")

    def log_detection(self, image_path, motion_area, species_name="Unknown species",
                     confidence_score=0.0, processing_time=0.0, api_success=False,
                     animals_detected=None, detection_count=None,
                     max_detection_confidence=None, contour_count=None,
                     largest_contour_area=None, foreground_pixel_count=None,
                     gate_would_suppress=None, background_drift=None,
                     detection_status=None, sharpness_score=None,
                     below_sharpness_floor=None, person_confidence=None,
                     top_species_raw=None, top_species_score=None,
                     scene_similarity=None, scene_gate_muted=None,
                     review_sampled_out=None, human_proximity_muted=None) -> Optional[int]:
        """Log a detection event to the database.

        The trailing keyword arguments are the Phase-1 richer-logging fields
        (ADR-004); all default to None so existing callers keep working.
        `hour_of_day` is derived from the insert time. `sharpness_score`,
        `below_sharpness_floor`, and `person_confidence` are Task 1's
        observability fields (already computed upstream, now persisted).
        `top_species_raw`/`top_species_score` are Task 3's: the classifier's
        raw top-1 prediction label/score, distinct from the (possibly
        rolled-up) ensemble `species_name`. `scene_similarity`/
        `scene_gate_muted` are Task 2's: the scene-unchanged gate's
        comparator score against the empty-scene reference set and whether
        it crossed the mute threshold. `review_sampled_out` is the
        REVIEW-channel sampling gate's decision; callers normally leave it
        None here (the detection id it's keyed on doesn't exist until this
        INSERT returns) and set it afterwards via update_review_sampled_out.
        `human_proximity_muted` is the human-proximity mute gate's decision
        (True/False for review-class bursts, None otherwise) — unlike
        `review_sampled_out` it doesn't depend on this row's own id, so it's
        set directly on the initial INSERT.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Use local wall-clock time explicitly. SQLite's
                # CURRENT_TIMESTAMP default is UTC; since image filenames are
                # also stamped with local time (datetime.now()), we must keep
                # them in sync by always writing the timestamp ourselves.
                now = datetime.now()
                # Column name -> value in one place so the INSERT column list and
                # the bound values can't drift out of sync.
                values = {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": str(image_path),
                    "motion_area": motion_area,
                    "species_name": species_name,
                    "confidence_score": confidence_score,
                    "processing_time": processing_time,
                    "api_success": api_success,
                    "animals_detected": animals_detected,
                    "detection_count": detection_count,
                    "max_detection_confidence": max_detection_confidence,
                    "contour_count": contour_count,
                    "largest_contour_area": largest_contour_area,
                    "foreground_pixel_count": foreground_pixel_count,
                    "hour_of_day": now.hour,
                    "gate_would_suppress": gate_would_suppress,
                    "background_drift": background_drift,
                    "detection_status": detection_status,
                    "sharpness_score": sharpness_score,
                    "below_sharpness_floor": below_sharpness_floor,
                    "person_confidence": person_confidence,
                    "top_species_raw": top_species_raw,
                    "top_species_score": top_species_score,
                    "scene_similarity": scene_similarity,
                    "scene_gate_muted": scene_gate_muted,
                    "review_sampled_out": review_sampled_out,
                    "human_proximity_muted": human_proximity_muted,
                }
                columns = ", ".join(values)
                placeholders = ", ".join("?" * len(values))
                cursor.execute(
                    f"INSERT INTO detections ({columns}) VALUES ({placeholders})",
                    tuple(values.values()),
                )

                detection_id = cursor.lastrowid

                # Update or insert species record
                if species_name != "Unknown species":
                    cursor.execute('''
                        INSERT OR IGNORE INTO species (name, first_detected, detection_count)
                        VALUES (?, ?, 0)
                    ''', (species_name, now.strftime("%Y-%m-%d %H:%M:%S")))

                    cursor.execute(
                        'UPDATE species SET detection_count = detection_count + 1 WHERE name = ?',
                        (species_name,)
                    )

                conn.commit()
                return detection_id

        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to log detection: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error logging detection: {e}") from e

    def update_review_sampled_out(self, detection_id: int, sampled_out: bool) -> None:
        """Set review_sampled_out on an already-inserted detection row.

        The REVIEW-channel sampling decision (see
        wildlife_system.is_review_sampled_out) is deterministic on
        detection_id, but detection_id — the autoincrement primary key — only
        exists once log_detection's INSERT has returned, so it can't be
        included in the original INSERT the way the other observability
        columns are. This is a small, targeted follow-up UPDATE instead.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE detections SET review_sampled_out = ? WHERE id = ?",
                    (sampled_out, detection_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to update review_sampled_out: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error updating review_sampled_out: {e}") from e

    def update_human_proximity_muted(self, detection_id: int, muted: bool) -> None:
        """Set human_proximity_muted on an already-inserted detection row.

        Modeled exactly on update_review_sampled_out: used by the deferred
        REVIEW-send gate (wildlife_system._deferred_review_send) to persist
        a cancel-on-human decision that's only known after the defer window
        has elapsed — unlike the synchronous human_proximity_muted value
        process_detection sets on the initial INSERT (which only looks
        BACKWARD from a prior HUMAN detection), this is the LEADING-EDGE
        case: a HUMAN-status detection landed shortly AFTER this burst.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE detections SET human_proximity_muted = ? WHERE id = ?",
                    (muted, detection_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to update human_proximity_muted: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error updating human_proximity_muted: {e}") from e

    # Human-tap labels map to these canonical strings (see feedback_protocol.py,
    # the single source of truth for wire codes; this tuple must cover every
    # CODE_TO_LABEL value, including legacy wrong_species, which is parse-only
    # on the keyboard but still a valid stored label for old channel messages).
    VALID_FEEDBACK_LABELS = (
        "animal", "animal_wrong_id", "person", "false_positive", "cant_tell",
        "wrong_species",
    )

    def add_feedback(self, detection_id: int, label: str,
                     source: str = "human") -> Optional[int]:
        """Append a ground-truth label for a detection (never overwrites).

        Returns the new feedback row id. Raises on an unknown label so a bad
        callback can't silently poison the corpus.
        """
        if label not in self.VALID_FEEDBACK_LABELS:
            raise DatabaseOperationError(
                f"Invalid feedback label '{label}'; expected one of {self.VALID_FEEDBACK_LABELS}"
            )
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Explicitly pass local time so we don't rely on SQLite's
                # CURRENT_TIMESTAMP default (which is always UTC).
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO detection_feedback (detection_id, label, source, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (detection_id, label, source, created_at))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to add feedback: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error adding feedback: {e}") from e

    def get_feedback(self, detection_id: int) -> List[tuple]:
        """Return all feedback rows for a detection, oldest first.

        Each row is (id, detection_id, label, source, created_at).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, detection_id, label, source, created_at
                    FROM detection_feedback
                    WHERE detection_id = ?
                    ORDER BY created_at ASC, id ASC
                ''', (detection_id,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get feedback: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting feedback: {e}") from e
    
    def get_recent_detections(self, limit=10) -> List[DetectionRecord]:
        """Get recent detection records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, timestamp, species_name, confidence_score, motion_area, 
                           image_path, processing_time, api_success
                    FROM detections
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                return [
                    DetectionRecord(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        species_name=row[2],
                        confidence_score=row[3],
                        motion_area=row[4],
                        image_path=row[5],
                        processing_time=row[6],
                        api_success=bool(row[7])
                    ) for row in rows
                ]
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get recent detections: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting detections: {e}") from e
    
    def get_species_stats(self):
        """Get species detection statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name, detection_count, first_detected
                    FROM species
                    ORDER BY detection_count DESC
                ''')
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting species stats: {e}", exc_info=True)
            return []
    
    def get_daily_detections(self, date=None):
        """Get detection count for a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM detections
                    WHERE DATE(timestamp) = ?
                ''', (date,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting daily detections: {e}", exc_info=True)
            return 0
    
    def cleanup_old_detections(self, days_to_keep=90):
        """Remove detection records older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use parameterized query to prevent SQL injection
                cursor.execute('''
                    DELETE FROM detections
                    WHERE timestamp < datetime('now', ? || ' days')
                ''', (f'-{days_to_keep}',))
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old detections (older than {days_to_keep} days)")
                return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old detections: {e}", exc_info=True)
            return 0
    
    def get_human_detections_older_than(self, cutoff: datetime) -> List[tuple]:
        """Return (id, image_path, timestamp) for HUMAN-status detections older than cutoff.

        Used by the Task 3 privacy purge: saved photos of human detections are
        deleted after `human_retention_hours`, but the DB row (a metadata-only
        record) is kept. `timestamp` is stored as a local wall-clock string in
        "%Y-%m-%d %H:%M:%S" format, which sorts lexicographically identically
        to chronological order, so a plain string comparison against the
        cutoff (formatted the same way) is correct here.
        """
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, image_path, timestamp
                    FROM detections
                    WHERE detection_status = 'human' AND timestamp < ?
                ''', (cutoff_str,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get human detections: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting human detections: {e}") from e

    def get_human_adjacent_review_detections(self, cutoff: datetime,
                                              window_seconds: float) -> List[tuple]:
        """Return (id, image_path, timestamp) for review-class rows that sit
        within `window_seconds` of a HUMAN-status detection, in EITHER
        direction, and are older than `cutoff`.

        Extends the Task 3 privacy purge (get_human_detections_older_than) to
        catch review-class (no_animal/unclassifiable) bursts that really
        contain a person but were misclassified: the human-proximity mute
        gate only looks BACKWARD from a HUMAN burst, so a burst just BEFORE
        the first HUMAN burst of a visit can leak a recognisable face to
        REVIEW and then sit on disk for the full ~300-burst rotation (see
        the 2026-07-31 fix, burst 3909, leaked 81s before the visit's first
        HUMAN burst). Modeled on get_human_detections_older_than; `timestamp`
        is stored as a local wall-clock string in "%Y-%m-%d %H:%M:%S" format.

        Perf note (2026-08-01): this was originally a single SQL query using
        a correlated EXISTS subquery with strftime('%s', ...) on both sides
        — SQLite can't use any index for that (it re-parses every timestamp
        pair), measured at 3.9s on the real 4162-row/279-match production DB.
        A JOIN + datetime(...,'+/-N seconds') rewrite still measured ~1.6s.
        Since purge_human_bursts() (and so this query) runs from
        cleanup_old_images() after every single detection on a Pi that's
        also running SpeciesNet inference, the match is now done in Python
        instead: two flat, indexed queries (the review-row range predicate
        uses idx_detections_timestamp; the HUMAN-row query has no WHERE and
        needs no index) followed by a single bisect per review row against
        the sorted, once-parsed list of HUMAN timestamps — O(R log H)
        instead of O(R * H) string/date reparsing in SQL.

        Malformed/NULL timestamp strings (in either query's rows) are
        skipped defensively during parsing rather than raising — a single
        bad row must not blow up the whole purge sweep. Returns []  when
        window_seconds <= 0 (disabled, the rollback lever) without touching
        the DB.
        """
        if window_seconds <= 0:
            return []
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Indexed range predicate (idx_detections_timestamp) — no
                # per-row string reparsing, unlike the old EXISTS subquery.
                cursor.execute('''
                    SELECT id, image_path, timestamp
                    FROM detections
                    WHERE detection_status IN ('no_animal', 'unclassifiable')
                      AND timestamp < ?
                ''', (cutoff_str,))
                review_rows = cursor.fetchall()

                # All HUMAN-status timestamps, parsed and sorted once so
                # every review row's nearest match is a single bisect away.
                cursor.execute('''
                    SELECT timestamp
                    FROM detections
                    WHERE detection_status = 'human'
                    ORDER BY timestamp ASC
                ''')
                human_ts_rows = cursor.fetchall()

            human_times = []
            for (ts_str,) in human_ts_rows:
                try:
                    human_times.append(
                        datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    )
                except (TypeError, ValueError):
                    # Malformed/NULL HUMAN timestamp — skip, don't crash the
                    # whole purge sweep over one bad row.
                    continue
            human_times.sort()

            if not human_times:
                return []

            matches = []
            for row_id, image_path, ts_str in review_rows:
                try:
                    review_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError):
                    # Malformed/NULL review-row timestamp — skip, same
                    # fail-safe as above.
                    continue

                # The nearest HUMAN timestamp to review_time is always the
                # insertion point or its immediate predecessor in the
                # sorted list — no need to scan every HUMAN timestamp.
                idx = bisect.bisect_left(human_times, review_time)
                is_adjacent = False
                if idx < len(human_times):
                    is_adjacent = abs(
                        (human_times[idx] - review_time).total_seconds()
                    ) <= window_seconds
                if not is_adjacent and idx > 0:
                    is_adjacent = abs(
                        (human_times[idx - 1] - review_time).total_seconds()
                    ) <= window_seconds

                if is_adjacent:
                    matches.append((row_id, image_path, ts_str))

            return matches
        except sqlite3.Error as e:
            raise DatabaseOperationError(
                f"Failed to get human-adjacent review detections: {e}"
            ) from e
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error getting human-adjacent review detections: {e}"
            ) from e

    def get_recent_review_detections(self, limit: int, max_age_hours: float) -> List[tuple]:
        """Return (image_path, timestamp) for recent review-class detections.

        Used to seed the scene-unchanged gate's rolling empty-scene reference
        set: review-class rows (`detection_status` in `no_animal` or
        `unclassifiable` — i.e. no animal was found) within the last
        `max_age_hours`, newest first, capped at `limit`. Modeled on
        `get_human_detections_older_than`; `timestamp` is stored as a local
        wall-clock string in "%Y-%m-%d %H:%M:%S" format, so a plain string
        comparison against a cutoff formatted the same way is correct here.
        """
        cutoff_str = (datetime.now() - timedelta(hours=max_age_hours)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT image_path, timestamp
                    FROM detections
                    WHERE detection_status IN ('no_animal', 'unclassifiable')
                      AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (cutoff_str, limit))
                return [
                    (row[0], datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S"))
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get recent review detections: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting recent review detections: {e}") from e

    def get_last_human_detection_time(self) -> Optional[datetime]:
        """Return the timestamp of the most recent HUMAN-status detection, or
        None if there is none.

        Used to seed the human-proximity mute gate's in-memory state
        (`WildlifeSystem._last_human_detection_at`) at startup, so a restart
        doesn't lose the look-back window. Modeled on
        `get_recent_review_detections`; `timestamp` is stored as a local
        wall-clock string in "%Y-%m-%d %H:%M:%S" format.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp
                    FROM detections
                    WHERE detection_status = 'human'
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''')
                row = cursor.fetchone()
                if row is None:
                    return None
                return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get last human detection time: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting last human detection time: {e}") from e

    def get_recent_human_detection_times(self, since: datetime) -> List[datetime]:
        """Return timestamps of all HUMAN-status detections at/after `since`.

        Used to seed the human-density condition's in-memory state
        (`WildlifeSystem._recent_human_detection_times`) at startup, so a
        restart doesn't lose track of an in-progress "garden is occupied"
        streak. Modeled on `get_last_human_detection_time`; `timestamp` is
        stored as a local wall-clock string in "%Y-%m-%d %H:%M:%S" format, so
        a plain string comparison against a cutoff formatted the same way is
        correct here. Returns oldest-first (no particular ordering is
        required by callers, but this matches insertion order).
        """
        since_str = since.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp
                    FROM detections
                    WHERE detection_status = 'human' AND timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (since_str,))
                return [
                    datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to get recent human detection times: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error getting recent human detection times: {e}") from e

    def is_first_detection_today(self, species_name):
        """Check if this is the first detection of this species today"""
        today = datetime.now().strftime('%Y-%m-%d')

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM detections
                    WHERE species_name = ? AND DATE(timestamp) = ?
                ''', (species_name, today))
                count = cursor.fetchone()[0]
                return count == 0  # True if no detections today
        except Exception as e:
            logger.error(f"Error checking first detection: {e}", exc_info=True)
            return False

