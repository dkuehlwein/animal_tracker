"""
Unit tests for database management system.
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import sys
sys.path.append('src')

from database_manager import (
    DatabaseManager, DatabaseError, DatabaseConnectionError, 
    DatabaseOperationError, DetectionRecord
)
from config import Config


class TestDetectionRecord:
    """Test detection record data class."""
    
    def test_detection_record_creation(self):
        """Test detection record creation with minimal data."""
        record = DetectionRecord(
            image_path="test/path.jpg",
            motion_area=1500
        )
        
        assert record.image_path == "test/path.jpg"
        assert record.motion_area == 1500
        assert record.species_name == "Unknown species"
        assert record.confidence_score == 0.0
        assert record.processing_time is None
        assert record.api_success is False
    
    def test_detection_record_full_data(self):
        """Test detection record creation with full data."""
        timestamp = datetime.now()
        record = DetectionRecord(
            image_path="test/hedgehog.jpg",
            motion_area=2500,
            species_name="European Hedgehog",
            confidence_score=0.87,
            processing_time=2.5,
            api_success=True,
            timestamp=timestamp
        )
        
        assert record.species_name == "European Hedgehog"
        assert record.confidence_score == 0.87
        assert record.processing_time == 2.5
        assert record.api_success is True
        assert record.timestamp == timestamp


class TestDatabaseManager:
    """Test database management functionality."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.config = Config.create_test_config()
        self.config.storage.database_path = self.db_path
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database and table creation."""
        db_manager = DatabaseManager(self.config)
        
        assert self.db_path.exists()
        
        # Check that tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'detections' in tables
            assert 'species' in tables
    
    def test_database_schema_validation(self):
        """Test database schema validation."""
        db_manager = DatabaseManager(self.config)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check detections table schema
            cursor.execute("PRAGMA table_info(detections)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            expected_columns = {
                'id': 'INTEGER',
                'timestamp': 'DATETIME',
                'image_path': 'TEXT',
                'motion_area': 'INTEGER',
                'species_name': 'TEXT',
                'confidence_score': 'REAL',
                'processing_time': 'REAL',
                'api_success': 'BOOLEAN'
            }
            
            for col_name, col_type in expected_columns.items():
                assert col_name in columns
                assert col_type in columns[col_name]
    
    def test_insert_detection_minimal(self):
        """Test inserting detection with minimal data."""
        db_manager = DatabaseManager(self.config)
        
        record = DetectionRecord(
            image_path="test/minimal.jpg",
            motion_area=1000
        )
        
        detection_id = db_manager.insert_detection(record)
        assert detection_id is not None
        assert detection_id > 0
        
        # Verify data was inserted correctly
        detections = db_manager.get_recent_detections(1)
        assert len(detections) == 1
        assert detections[0]['image_path'] == "test/minimal.jpg"
        assert detections[0]['motion_area'] == 1000
        assert detections[0]['species_name'] == "Unknown species"
    
    def test_insert_detection_full_data(self):
        """Test inserting detection with complete data."""
        db_manager = DatabaseManager(self.config)
        
        timestamp = datetime.now()
        record = DetectionRecord(
            image_path="test/hedgehog.jpg",
            motion_area=2500,
            species_name="European Hedgehog",
            confidence_score=0.87,
            processing_time=2.5,
            api_success=True,
            timestamp=timestamp
        )
        
        detection_id = db_manager.insert_detection(record)
        assert detection_id is not None
        
        # Verify full data
        detections = db_manager.get_recent_detections(1)
        detection = detections[0]
        
        assert detection['species_name'] == "European Hedgehog"
        assert detection['confidence_score'] == 0.87
        assert detection['processing_time'] == 2.5
        assert detection['api_success'] == 1  # SQLite boolean as integer
    
    def test_update_species_tracking(self):
        """Test species tracking functionality."""
        db_manager = DatabaseManager(self.config)
        
        # Insert first detection of a species
        record = DetectionRecord(
            image_path="test/hedgehog1.jpg",
            motion_area=2000,
            species_name="European Hedgehog"
        )
        db_manager.insert_detection(record)
        
        # Insert second detection of same species
        record2 = DetectionRecord(
            image_path="test/hedgehog2.jpg",
            motion_area=2200,
            species_name="European Hedgehog"
        )
        db_manager.insert_detection(record2)
        
        # Check species tracking
        species_stats = db_manager.get_species_stats()
        assert len(species_stats) >= 1
        
        hedgehog_stats = next((s for s in species_stats if s['name'] == 'European Hedgehog'), None)
        assert hedgehog_stats is not None
        assert hedgehog_stats['detection_count'] == 2
    
    def test_get_recent_detections(self):
        """Test retrieving recent detections."""
        db_manager = DatabaseManager(self.config)
        
        # Insert multiple detections with different times
        base_time = datetime.now()
        for i in range(5):
            record = DetectionRecord(
                image_path=f"test/detection_{i}.jpg",
                motion_area=1000 + i * 100,
                timestamp=base_time - timedelta(minutes=i)
            )
            db_manager.insert_detection(record)
        
        # Get recent detections
        recent = db_manager.get_recent_detections(3)
        assert len(recent) == 3
        
        # Should be ordered by timestamp descending (most recent first)
        for i in range(len(recent) - 1):
            current_time = datetime.fromisoformat(recent[i]['timestamp'])
            next_time = datetime.fromisoformat(recent[i + 1]['timestamp'])
            assert current_time >= next_time
    
    def test_get_detections_by_species(self):
        """Test filtering detections by species."""
        db_manager = DatabaseManager(self.config)
        
        # Insert detections for different species
        species_data = [
            ("European Hedgehog", 3),
            ("Red Fox", 2),
            ("Unknown species", 1)
        ]
        
        for species, count in species_data:
            for i in range(count):
                record = DetectionRecord(
                    image_path=f"test/{species}_{i}.jpg",
                    motion_area=1500,
                    species_name=species
                )
                db_manager.insert_detection(record)
        
        # Test filtering
        hedgehog_detections = db_manager.get_detections_by_species("European Hedgehog")
        assert len(hedgehog_detections) == 3
        
        fox_detections = db_manager.get_detections_by_species("Red Fox")
        assert len(fox_detections) == 2
        
        unknown_detections = db_manager.get_detections_by_species("Unknown species")
        assert len(unknown_detections) == 1
    
    def test_get_detection_statistics(self):
        """Test detection statistics calculation."""
        db_manager = DatabaseManager(self.config)
        
        # Insert test data
        base_time = datetime.now()
        test_data = [
            ("European Hedgehog", 0.85, 2.5, True),
            ("European Hedgehog", 0.92, 1.8, True),
            ("Red Fox", 0.78, 3.2, True),
            ("Unknown species", 0.0, 0.0, False),
        ]
        
        for species, confidence, proc_time, api_success in test_data:
            record = DetectionRecord(
                image_path=f"test/{species}_test.jpg",
                motion_area=2000,
                species_name=species,
                confidence_score=confidence,
                processing_time=proc_time,
                api_success=api_success,
                timestamp=base_time - timedelta(minutes=len(test_data))
            )
            db_manager.insert_detection(record)
        
        # Get statistics
        stats = db_manager.get_detection_statistics()
        
        assert stats['total_detections'] == 4
        assert stats['successful_identifications'] == 3
        assert stats['unique_species'] == 3  # Includes "Unknown species"
        assert stats['average_confidence'] > 0
        assert stats['average_processing_time'] > 0
    
    def test_cleanup_old_detections(self):
        """Test cleanup of old detections."""
        db_manager = DatabaseManager(self.config)
        
        # Insert old and new detections
        old_time = datetime.now() - timedelta(days=40)
        recent_time = datetime.now() - timedelta(days=5)
        
        for i in range(3):
            # Old detection
            old_record = DetectionRecord(
                image_path=f"test/old_{i}.jpg",
                motion_area=1000,
                timestamp=old_time - timedelta(hours=i)
            )
            db_manager.insert_detection(old_record)
            
            # Recent detection
            recent_record = DetectionRecord(
                image_path=f"test/recent_{i}.jpg",
                motion_area=1000,
                timestamp=recent_time - timedelta(hours=i)
            )
            db_manager.insert_detection(recent_record)
        
        # Clean up detections older than 30 days
        cleaned_count = db_manager.cleanup_old_detections(days=30)
        assert cleaned_count == 3
        
        # Verify only recent detections remain
        remaining = db_manager.get_recent_detections(10)
        assert len(remaining) == 3
        assert all('recent' in detection['image_path'] for detection in remaining)
    
    def test_database_connection_error(self):
        """Test database connection error handling."""
        # Use invalid database path
        invalid_config = Config.create_test_config()
        invalid_config.storage.database_path = Path("/invalid/path/db.sqlite")
        
        with pytest.raises(DatabaseConnectionError, match="Failed to connect to database"):
            DatabaseManager(invalid_config)
    
    def test_database_operation_error(self):
        """Test database operation error handling."""
        db_manager = DatabaseManager(self.config)
        
        # Corrupt the database by writing invalid data directly
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE detections")
        
        record = DetectionRecord(
            image_path="test/error.jpg",
            motion_area=1000
        )
        
        with pytest.raises(DatabaseOperationError, match="Failed to insert detection"):
            db_manager.insert_detection(record)
    
    @patch('sqlite3.connect')
    def test_database_transaction_rollback(self, mock_connect):
        """Test database transaction rollback on error."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Simulate error during transaction
        mock_cursor.execute.side_effect = sqlite3.Error("Transaction failed")
        
        db_manager = DatabaseManager(self.config)
        
        record = DetectionRecord(
            image_path="test/rollback.jpg",
            motion_area=1000
        )
        
        with pytest.raises(DatabaseOperationError):
            db_manager.insert_detection(record)
        
        # Verify rollback was called
        mock_conn.rollback.assert_called_once()
    
    def test_concurrent_database_access(self):
        """Test concurrent database access safety."""
        import threading
        import time
        
        db_manager = DatabaseManager(self.config)
        
        results = []
        errors = []
        
        def insert_detections(thread_id):
            try:
                for i in range(10):
                    record = DetectionRecord(
                        image_path=f"test/thread_{thread_id}_{i}.jpg",
                        motion_area=1000 + i,
                        species_name=f"Species_{thread_id}"
                    )
                    detection_id = db_manager.insert_detection(record)
                    results.append(detection_id)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=insert_detections, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and all insertions succeeded
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 30  # 3 threads × 10 insertions
        assert all(result is not None for result in results)
    
    def test_database_backup_and_restore(self, tmp_path):
        """Test database backup functionality."""
        db_manager = DatabaseManager(self.config)
        
        # Insert some test data
        for i in range(3):
            record = DetectionRecord(
                image_path=f"test/backup_{i}.jpg",
                motion_area=1500 + i * 100
            )
            db_manager.insert_detection(record)
        
        # Create backup
        backup_path = tmp_path / "backup.db"
        success = db_manager.create_backup(backup_path)
        assert success
        assert backup_path.exists()
        
        # Verify backup contains same data
        backup_manager = DatabaseManager(self.config)
        backup_manager.config.storage.database_path = backup_path
        backup_manager._connect()
        
        original_detections = db_manager.get_recent_detections(10)
        backup_detections = backup_manager.get_recent_detections(10)
        
        assert len(original_detections) == len(backup_detections)
        assert len(backup_detections) == 3


if __name__ == '__main__':
    pytest.main([__file__])