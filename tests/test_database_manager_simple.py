"""
Simple tests for database management system based on actual implementation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.append('src')

from database_manager import DatabaseManager
from config import Config


class TestDatabaseManager:
    """Test actual database manager functionality."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        
        # Create config with custom database path using environment override
        import os
        original_path = os.environ.get('DATABASE_PATH')
        os.environ['DATABASE_PATH'] = str(self.db_path)
        os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'
        os.environ['TELEGRAM_CHAT_ID'] = 'test_chat'
        
        try:
            self.config = Config()
        finally:
            # Restore original environment
            if original_path:
                os.environ['DATABASE_PATH'] = original_path
            elif 'DATABASE_PATH' in os.environ:
                del os.environ['DATABASE_PATH']
        
        # Create database manager
        self.db_manager = DatabaseManager(self.config)
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database initialization creates the file."""
        assert self.db_path.exists()
    
    def test_log_detection_basic(self):
        """Test basic detection logging."""
        detection_id = self.db_manager.log_detection(
            image_path="test/image.jpg",
            motion_area=1500
        )
        
        assert detection_id is not None
        assert detection_id > 0
    
    def test_log_detection_with_species(self):
        """Test detection logging with species information."""
        detection_id = self.db_manager.log_detection(
            image_path="test/hedgehog.jpg",
            motion_area=2500,
            species_name="European Hedgehog",
            confidence_score=0.87,
            processing_time=2.5,
            api_success=True
        )
        
        assert detection_id is not None
    
    def test_get_recent_detections(self):
        """Test retrieving recent detections."""
        # Log some detections
        self.db_manager.log_detection("test1.jpg", 1000)
        self.db_manager.log_detection("test2.jpg", 2000)
        self.db_manager.log_detection("test3.jpg", 3000)
        
        recent = self.db_manager.get_recent_detections(2)
        
        assert len(recent) == 2
        # Should be ordered by timestamp DESC (most recent first)
        assert recent[0][4] == "test3.jpg"  # image_path is index 4
    
    def test_get_species_stats(self):
        """Test species statistics."""
        # Log some detections with species
        self.db_manager.log_detection("hedgehog1.jpg", 2000, "European Hedgehog")
        self.db_manager.log_detection("hedgehog2.jpg", 2200, "European Hedgehog")
        self.db_manager.log_detection("fox1.jpg", 1800, "Red Fox")
        
        stats = self.db_manager.get_species_stats()
        
        assert len(stats) >= 2
        # Should be ordered by detection count DESC
        assert stats[0][1] >= stats[1][1]  # detection_count is index 1
    
    def test_get_daily_detections(self):
        """Test daily detection count."""
        # Log a detection
        self.db_manager.log_detection("today.jpg", 1500)
        
        # Check today's count
        today_count = self.db_manager.get_daily_detections()
        assert today_count >= 1
    
    def test_cleanup_old_detections(self):
        """Test cleanup functionality."""
        # Log some detections
        for i in range(5):
            self.db_manager.log_detection(f"test{i}.jpg", 1000 + i * 100)
        
        # Clean up (should not delete recent ones)
        deleted = self.db_manager.cleanup_old_detections(days_to_keep=1)
        
        # Should return number of deleted records (likely 0 for recent data)
        assert isinstance(deleted, int)
        assert deleted >= 0
    
    def test_is_first_detection_today(self):
        """Test first detection check."""
        species = "European Hedgehog"
        
        # Should be first detection
        assert self.db_manager.is_first_detection_today(species) is True
        
        # Log a detection
        self.db_manager.log_detection("hedgehog.jpg", 2000, species)
        
        # Should no longer be first detection
        assert self.db_manager.is_first_detection_today(species) is False


if __name__ == '__main__':
    pytest.main([__file__])