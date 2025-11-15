"""
Simple functionality tests that work with actual implementation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append('src')

from species_identifier import SpeciesIdentifier
from config import Config


class TestBasicFunctionality:
    """Test basic functionality that actually works."""
    
    def test_config_creation(self):
        """Test that config can be created."""
        config = Config.create_test_config()
        
        assert config.telegram_token == 'test_token'
        assert config.telegram_chat_id == 'test_chat'
        assert config.camera.main_resolution == (1920, 1080)
        assert config.motion.motion_threshold == 2000
    
    def test_species_identifier_basic(self):
        """Test species identifier basic functionality."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config)

        # Test initialization (lazy-loaded)
        assert identifier._model is None
        assert identifier._model_loaded is False
        assert identifier.config is not None
    
    def test_species_identification_with_real_identifier(self):
        """Test species identification with real SpeciesIdentifier."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config)

        # Create temporary image file
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_image = temp_dir / "test.jpg"
            test_image.write_bytes(b"fake_image")

            # Test identification (will fail without SpeciesNet but returns IdentificationResult)
            result = identifier.identify_species(str(test_image))

            # Verify IdentificationResult structure
            assert hasattr(result, 'species_name')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'api_success')
            assert hasattr(result, 'processing_time')
            assert hasattr(result, 'fallback_reason')

            # Check result values
            assert isinstance(result.species_name, str)
            assert isinstance(result.confidence, float)
            assert result.processing_time >= 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_database_manager_basic(self):
        """Test database manager basic functionality."""
        # Use default test config and let it create database in default location
        config = Config.create_test_config()
        
        # Create data directory if it doesn't exist
        config.storage.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and test database manager
        from database_manager import DatabaseManager
        db_manager = DatabaseManager(config)
        
        # Test basic logging
        detection_id = db_manager.log_detection("test.jpg", 1500)
        assert detection_id is not None
        
        # Test recent detections
        recent = db_manager.get_recent_detections(1)
        assert len(recent) >= 1
        
        # Test cleanup
        cleaned = db_manager.cleanup_old_detections(days_to_keep=30)
        assert isinstance(cleaned, int)
        
        # Cleanup test database
        db_path = Path(config.storage.database_path)
        if db_path.exists():
            db_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])