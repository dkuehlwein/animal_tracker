import sqlite3
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from config import Config

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseOperationError(DatabaseError):
    """Raised when database operations fail."""
    pass


@dataclass
class DetectionRecord:
    """Data class for detection records."""
    id: Optional[int]
    timestamp: datetime
    image_path: str
    motion_area: int
    species_name: str = "Unknown species"
    confidence_score: float = 0.0
    processing_time: float = 0.0
    api_success: bool = False

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        # Ensure database path is absolute to avoid path resolution issues
        self.db_path = str(Path(config.storage.database_path).resolve())
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            # Ensure data directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
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
                
                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"Failed to initialize database: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error initializing database: {e}") from e
    
    def log_detection(self, image_path, motion_area, species_name="Unknown species", 
                     confidence_score=0.0, processing_time=0.0, api_success=False) -> Optional[int]:
        """Log a detection event to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert detection record
                cursor.execute('''
                    INSERT INTO detections 
                    (image_path, motion_area, species_name, confidence_score, 
                     processing_time, api_success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (str(image_path), motion_area, species_name, 
                      confidence_score, processing_time, api_success))
                
                detection_id = cursor.lastrowid
                
                # Update or insert species record
                if species_name != "Unknown species":
                    cursor.execute('''
                        INSERT OR IGNORE INTO species (name, first_detected, detection_count)
                        VALUES (?, ?, 1)
                    ''', (species_name, datetime.now()))
                    
                    cursor.execute('''
                        UPDATE species SET detection_count = detection_count + 1
                        WHERE name = ? AND EXISTS (
                            SELECT 1 FROM species WHERE name = ? AND detection_count > 0
                        )
                    ''', (species_name, species_name))
                
                conn.commit()
                return detection_id
                
        except sqlite3.Error as e:
            raise DatabaseOperationError(f"Failed to log detection: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error logging detection: {e}") from e
    
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