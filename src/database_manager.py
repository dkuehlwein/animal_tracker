import sqlite3
import logging
from datetime import datetime
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
                     gate_would_suppress=None, background_drift=None) -> Optional[int]:
        """Log a detection event to the database.

        The trailing keyword arguments are the Phase-1 richer-logging fields
        (ADR-004); all default to None so existing callers keep working.
        `hour_of_day` is derived from the insert time.
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

    # Human-tap labels map to these canonical strings (see telegram_feedback.py).
    VALID_FEEDBACK_LABELS = ("animal", "false_positive", "wrong_species")

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

