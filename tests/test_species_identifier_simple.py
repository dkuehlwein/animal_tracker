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
        """Test identifier initializes with mock species list."""
        assert len(self.identifier.mock_species) > 0
        assert "European Hedgehog" in self.identifier.mock_species
        assert "Red Fox" in self.identifier.mock_species
    
    def test_identify_species_with_valid_image(self):
        """Test species identification with valid image file."""
        # Create a dummy image file
        test_image = self.temp_dir / "test_image.jpg"
        test_image.write_bytes(b"fake_image_data")
        
        result = self.identifier.identify_species(str(test_image))
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'species_name' in result
        assert 'confidence' in result
        assert 'api_success' in result
        assert 'processing_time' in result
        assert 'fallback_reason' in result
        
        # Check result values
        assert isinstance(result['species_name'], str)
        assert isinstance(result['confidence'], float)
        assert result['api_success'] is False  # Mock always returns False
        assert result['processing_time'] > 0
        assert result['fallback_reason'] == 'Mock implementation'
    
    def test_identify_species_with_nonexistent_image(self):
        """Test species identification with nonexistent image."""
        result = self.identifier.identify_species("nonexistent.jpg")
        
        assert result['species_name'] == 'Unknown species'
        assert result['confidence'] == 0.0
        assert result['api_success'] is False
        assert 'Image file not found' in result['fallback_reason']
    
    def test_mock_identification_logic(self):
        """Test that mock identification returns reasonable results."""
        # Create test image
        test_image = self.temp_dir / "test.jpg"
        test_image.write_bytes(b"test")
        
        # Run identification multiple times to test randomness
        results = []
        for _ in range(20):
            result = self.identifier.identify_species(str(test_image))
            results.append(result)
        
        # Should have some variety in results
        species_names = [r['species_name'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Should include "Unknown species" frequently (70% chance)
        unknown_count = species_names.count('Unknown species')
        assert unknown_count >= 10  # At least half should be unknown
        
        # Confidence should be 0 for unknown species
        for result in results:
            if result['species_name'] == 'Unknown species':
                assert result['confidence'] == 0.0
            else:
                assert result['confidence'] > 0.0
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.identifier.health_check()
        
        assert isinstance(health, dict)
        assert health['available'] is True
        assert 'service' in health
        assert 'version' in health
        assert 'supported_formats' in health
        assert 'jpg' in health['supported_formats']
    
    def test_get_supported_species(self):
        """Test getting supported species list."""
        species_list = self.identifier.get_supported_species()
        
        assert isinstance(species_list, list)
        assert len(species_list) > 0
        assert "European Hedgehog" in species_list
        assert "Red Fox" in species_list
        
        # Should be a copy, not the original
        species_list.append("Test Species")
        assert "Test Species" not in self.identifier.mock_species
    
    def test_get_statistics(self):
        """Test getting service statistics."""
        stats = self.identifier.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_identifications' in stats
        assert 'success_rate' in stats
        assert 'average_confidence' in stats
        assert 'most_common_species' in stats
        assert 'service_uptime' in stats
        
        assert isinstance(stats['total_identifications'], int)
        assert isinstance(stats['success_rate'], float)
        assert 0 <= stats['success_rate'] <= 1


if __name__ == '__main__':
    pytest.main([__file__])