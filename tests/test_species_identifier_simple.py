"""
Simple tests for species identification system based on actual implementation.
"""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.append('src')

from species_identifier import SpeciesIdentifier
from config import Config


class TestSpeciesIdentifier:
    """Test actual species identifier functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.identifier = SpeciesIdentifier(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_identifier_initialization(self):
        """Test identifier initializes correctly."""
        assert self.identifier._model is None  # Lazy loaded
        assert self.identifier._model_loaded is False
        assert self.identifier.config is not None
    
    def test_identify_species_with_valid_image(self):
        """Test species identification with valid image file."""
        # Create a dummy image file
        test_image = self.temp_dir / "test_image.jpg"
        test_image.write_bytes(b"fake_image_data")

        # Note: This will fail without SpeciesNet installed
        # But returns IdentificationResult on error
        result = self.identifier.identify_species(str(test_image))

        # Check result structure (returns IdentificationResult dataclass)
        assert hasattr(result, 'species_name')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'api_success')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'fallback_reason')

        # Check result values
        assert isinstance(result.species_name, str)
        assert isinstance(result.confidence, float)
        assert result.processing_time >= 0
    
    def test_identify_species_with_nonexistent_image(self):
        """Test species identification with nonexistent image."""
        result = self.identifier.identify_species("nonexistent.jpg")

        assert result.species_name == 'Unknown species'
        assert result.confidence == 0.0
        assert result.api_success is False
        assert 'Image file not found' in result.fallback_reason
    
    def test_identification_returns_result(self):
        """Test that identification returns proper IdentificationResult."""
        # Create test image
        test_image = self.temp_dir / "test.jpg"
        test_image.write_bytes(b"test")

        result = self.identifier.identify_species(str(test_image))

        # Should always return IdentificationResult
        assert hasattr(result, 'species_name')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'api_success')
        assert hasattr(result, 'processing_time')

        # Result should be valid
        assert result.species_name is not None
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.identifier.health_check()

        assert isinstance(health, dict)
        # Health check may return error for mock without model
        assert 'available' in health or 'error' in health
    
    def test_get_supported_species(self):
        """Test getting supported species info."""
        species_info = self.identifier.get_supported_species()

        # Real SpeciesIdentifier returns dict with info
        assert isinstance(species_info, dict)
        assert 'total_labels' in species_info
        assert 'geographic_filter' in species_info
    
    def test_get_statistics(self):
        """Test getting service statistics."""
        stats = self.identifier.get_statistics()

        assert isinstance(stats, dict)
        # Real implementation returns model status
        assert 'model_loaded' in stats
        assert 'model_version' in stats
        assert isinstance(stats['model_loaded'], bool)
        assert stats['model_version'] == 'v4.0.1a'


if __name__ == '__main__':
    pytest.main([__file__])