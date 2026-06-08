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

    def test_database_species_count_accuracy(self):
        """Test that species detection count is accurate (no off-by-one error)."""
        import sqlite3
        from database_manager import DatabaseManager

        # Use test config with clean database
        config = Config.create_test_config()
        config.storage.data_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any existing test database
        db_path = Path(config.storage.database_path)
        if db_path.exists():
            db_path.unlink()

        db_manager = DatabaseManager(config)

        # First detection of a new species
        db_manager.log_detection("test1.jpg", 1500, species_name="Red Fox")

        # Query species table directly
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT detection_count FROM species WHERE name = ?", ("Red Fox",))
            count = cursor.fetchone()[0]

        # After first detection, count should be 1 (not 2)
        assert count == 1, f"First detection should have count=1, got {count}"

        # Second detection of the same species
        db_manager.log_detection("test2.jpg", 1600, species_name="Red Fox")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT detection_count FROM species WHERE name = ?", ("Red Fox",))
            count = cursor.fetchone()[0]

        # After second detection, count should be 2
        assert count == 2, f"Second detection should have count=2, got {count}"

        # Third detection
        db_manager.log_detection("test3.jpg", 1700, species_name="Red Fox")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT detection_count FROM species WHERE name = ?", ("Red Fox",))
            count = cursor.fetchone()[0]

        # After third detection, count should be 3
        assert count == 3, f"Third detection should have count=3, got {count}"

        # Cleanup test database
        if db_path.exists():
            db_path.unlink()


class TestWildlifeSystemCaptionBuilder:
    """Test caption building in WildlifeSystem."""

    def test_build_caption_with_species_detected(self):
        """Test caption shows species when animals_detected is True."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        # Simulate a species result with animals detected
        detection_result = DetectionResult(
            animals_detected=True,
            detection_count=1,
            bounding_boxes=[{'confidence': 0.75, 'bbox': [0.1, 0.2, 0.3, 0.4]}],
            detections=[{'category': '1', 'conf': 0.75}],
            processing_time=1.5
        )

        species_result = {
            'species_name': 'aba05f1e;mammalia;rodentia;sciuridae;sciurus;vulgaris;eurasian red squirrel',
            'confidence': 0.85,
            'animals_detected': True,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=1500,
            timestamp=datetime(2026, 1, 17, 10, 30, 0),
            temperature=22.5
        )

        # Should show species info, not just "Motion detected"
        assert 'Eurasian Red Squirrel' in caption
        assert '85%' in caption
        assert 'Motion detected' not in caption

    def test_build_caption_no_animals_detected(self):
        """Test caption shows motion only when animals_detected is False."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        # Simulate result with no animals detected
        detection_result = DetectionResult(
            animals_detected=False,
            detection_count=0,
            bounding_boxes=[],
            detections=[],
            processing_time=1.0
        )

        species_result = {
            'species_name': 'Unknown species',
            'confidence': 0.0,
            'animals_detected': False,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=800,
            timestamp=datetime(2026, 1, 17, 10, 30, 0),
            temperature=20.0
        )

        # Should show motion detected
        assert 'Motion detected' in caption
        assert '800 px' in caption
        # Should also surface the classifier verdict
        assert 'Classifier' in caption
        assert 'Unknown species' in caption

    def test_build_caption_no_animals_includes_classifier_verdict(self):
        """No-animal caption must include a classifier line so 'Wrong species' is judgeable."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        detection_result = DetectionResult(
            animals_detected=False,
            detection_count=0,
            bounding_boxes=[],
            detections=[],
            processing_time=1.0,
        )
        species_result = {
            'species_name': 'Unknown species',
            'confidence': 0.0,
            'animals_detected': False,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=600,
            timestamp=datetime(2026, 3, 1, 8, 0, 0),
        )

        assert 'Motion detected' in caption
        assert 'Classifier' in caption

    def test_build_caption_blank_taxonomy_renders_cleanly(self):
        """Raw taxonomy '...;;;;;;blank' should appear as 'Blank', not the UUID string."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        # Typical "blank" result from SpeciesNet ensemble when animals_detected=True
        detection_result = DetectionResult(
            animals_detected=True,
            detection_count=1,
            bounding_boxes=[{'confidence': 0.6, 'bbox': [0, 0, 1, 1]}],
            detections=[],
            processing_time=1.0,
        )
        raw_blank = 'aba05f1e-dead-beef-0000-000000000000;;;;;;blank'
        species_result = {
            'species_name': raw_blank,
            'confidence': 0.72,
            'animals_detected': True,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=900,
            timestamp=datetime(2026, 3, 1, 8, 0, 0),
        )

        assert 'Blank' in caption
        assert 'aba05f1e' not in caption  # raw UUID must never appear

    def test_build_caption_human_taxonomy_renders_cleanly(self):
        """Raw 'human' taxonomy string should surface as 'Human' in the caption."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        detection_result = DetectionResult(
            animals_detected=True,
            detection_count=1,
            bounding_boxes=[{'confidence': 0.88, 'bbox': [0, 0, 1, 1]}],
            detections=[],
            processing_time=1.0,
        )
        raw_human = 'deadbeef-0000-0000-0000-000000000000;mammalia;primates;hominidae;homo;sapiens;human'
        species_result = {
            'species_name': raw_human,
            'confidence': 0.88,
            'animals_detected': True,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=1200,
            timestamp=datetime(2026, 3, 1, 8, 0, 0),
        )

        assert 'Human' in caption
        assert 'deadbeef' not in caption

    def test_build_caption_animal_still_shows_species_line(self):
        """Animal-detected branch must still show the species line (regression guard)."""
        from wildlife_system import WildlifeSystem
        from data_models import DetectionResult
        from datetime import datetime

        system = WildlifeSystem()

        detection_result = DetectionResult(
            animals_detected=True,
            detection_count=1,
            bounding_boxes=[{'confidence': 0.80, 'bbox': [0.1, 0.1, 0.5, 0.5]}],
            detections=[],
            processing_time=1.0,
        )
        species_result = {
            'species_name': 'Fox',
            'confidence': 0.91,
            'animals_detected': True,
            'detection_result': detection_result,
            'metadata': {},
        }

        caption = system._build_caption(
            species_result,
            motion_area=3000,
            timestamp=datetime(2026, 3, 1, 20, 15, 0),
            temperature=18.0,
        )

        assert 'Fox' in caption
        assert '91%' in caption
        assert 'Motion detected' not in caption

    def test_extract_species_name_empty_common_name_falls_back_to_scientific(self):
        """When common_name token is absent, fall back to genus+species."""
        from wildlife_system import WildlifeSystem

        system = WildlifeSystem()

        # Taxonomy with empty common_name but valid genus+species
        raw = 'uuid;mammalia;rodentia;sciuridae;sciurus;vulgaris;'
        result = system._extract_species_name(raw)
        assert result == 'Sciurus Vulgaris'

    def test_extract_species_name_none_input(self):
        """None/empty input must not raise and should return 'Unknown species'."""
        from wildlife_system import WildlifeSystem

        system = WildlifeSystem()

        assert system._extract_species_name('') == 'Unknown species'

    def test_extract_species_name_blank_token(self):
        """';;;;;;blank' should return 'Blank'."""
        from wildlife_system import WildlifeSystem

        system = WildlifeSystem()

        raw = 'some-uuid;;;;;;blank'
        assert system._extract_species_name(raw) == 'Blank'

    def test_extract_species_name_no_cv_result(self):
        """'no cv result' token should be returned readable (title-cased)."""
        from wildlife_system import WildlifeSystem

        system = WildlifeSystem()

        raw = 'some-uuid;;;;;;no cv result'
        assert system._extract_species_name(raw) == 'No Cv Result'


if __name__ == '__main__':
    pytest.main([__file__])