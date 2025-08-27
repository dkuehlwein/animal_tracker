"""
Unit tests for species identification system.
"""

import pytest
import numpy as np
import time
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import sys
sys.path.append('src')

from species_identifier import (
    SpeciesIdentifier, MockSpeciesIdentifier, SpeciesIdentificationError,
    IdentificationTimeout, IdentificationResult
)
from config import Config


class TestIdentificationResult:
    """Test identification result data class."""
    
    def test_identification_result_success(self):
        """Test successful identification result."""
        result = IdentificationResult(
            species_name="European Hedgehog",
            confidence=0.87,
            processing_time=2.5,
            api_success=True
        )
        
        assert result.species_name == "European Hedgehog"
        assert result.confidence == 0.87
        assert result.processing_time == 2.5
        assert result.api_success is True
        assert result.fallback_reason is None
    
    def test_identification_result_failure(self):
        """Test failed identification result."""
        result = IdentificationResult(
            species_name="Unknown species",
            confidence=0.0,
            processing_time=0.1,
            api_success=False,
            fallback_reason="Mock implementation"
        )
        
        assert result.species_name == "Unknown species"
        assert result.confidence == 0.0
        assert result.api_success is False
        assert result.fallback_reason == "Mock implementation"
    
    def test_identification_result_dict_conversion(self):
        """Test conversion to dictionary."""
        result = IdentificationResult(
            species_name="Red Fox",
            confidence=0.92,
            processing_time=1.8,
            api_success=True
        )
        
        result_dict = result.to_dict()
        expected_keys = {'species_name', 'confidence', 'processing_time', 'api_success', 'fallback_reason'}
        
        assert set(result_dict.keys()) == expected_keys
        assert result_dict['species_name'] == "Red Fox"
        assert result_dict['confidence'] == 0.92
        assert result_dict['api_success'] is True


class TestMockSpeciesIdentifier:
    """Test mock species identification implementation."""
    
    def test_mock_identifier_initialization(self):
        """Test mock identifier initialization."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        
        assert identifier.config == config
        assert not identifier.is_available()
    
    def test_mock_identifier_startup(self):
        """Test mock identifier startup process."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        
        identifier.start()
        assert identifier.is_available()
        
        identifier.stop()
        assert not identifier.is_available()
    
    def test_mock_identification_from_array(self):
        """Test identification from numpy array."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        identifier.start()
        
        # Create test image array
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result = identifier.identify_from_array(test_image)
        
        assert isinstance(result, IdentificationResult)
        assert result.species_name == "Unknown species"
        assert result.confidence == 0.0
        assert result.api_success is False
        assert result.fallback_reason == "Mock implementation"
        assert result.processing_time > 0
        
        identifier.stop()
    
    def test_mock_identification_from_file(self, tmp_path):
        """Test identification from image file."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        identifier.start()
        
        # Create a test image file (just a dummy file for testing)
        test_image_path = tmp_path / "test_image.jpg"
        test_image_path.write_bytes(b"fake_image_data")
        
        result = identifier.identify_from_file(test_image_path)
        
        assert isinstance(result, IdentificationResult)
        assert result.species_name == "Unknown species"
        assert result.confidence == 0.0
        assert result.api_success is False
        
        identifier.stop()
    
    def test_mock_identification_when_stopped(self):
        """Test identification when service is stopped."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = identifier.identify_from_array(test_image)
        
        assert result.species_name == "Unknown species"
        assert result.api_success is False
        assert result.fallback_reason == "Service not available"
    
    def test_mock_processing_delay_simulation(self):
        """Test mock processing delay simulation."""
        config = Config.create_test_config()
        # Set longer processing delay for testing
        config.performance.api_timeout = 5.0
        
        identifier = MockSpeciesIdentifier(config)
        identifier.start()
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = identifier.identify_from_array(test_image)
        end_time = time.time()
        
        # Should have some processing delay (mock adds 0.1-0.3s)
        assert result.processing_time >= 0.1
        assert end_time - start_time >= 0.1
        
        identifier.stop()
    
    def test_mock_stats_tracking(self):
        """Test statistics tracking in mock identifier."""
        config = Config.create_test_config()
        identifier = MockSpeciesIdentifier(config)
        identifier.start()
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Perform multiple identifications
        for _ in range(3):
            identifier.identify_from_array(test_image)
        
        stats = identifier.get_stats()
        
        assert stats['total_requests'] == 3
        assert stats['successful_requests'] == 0  # Mock always fails
        assert stats['average_processing_time'] > 0
        assert stats['is_available'] is True
        
        identifier.stop()


class TestSpeciesIdentifier:
    """Test high-level species identifier interface."""
    
    def test_species_identifier_with_mock(self):
        """Test species identifier with mock implementation."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        
        assert isinstance(identifier._implementation, MockSpeciesIdentifier)
    
    def test_species_identifier_context_manager(self):
        """Test species identifier as context manager."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        
        with identifier:
            assert identifier.is_operational()
            
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = identifier.identify_from_array(test_image)
            
            assert isinstance(result, IdentificationResult)
        
        # Should be stopped after context
        assert not identifier.is_operational()
    
    def test_species_identifier_batch_processing(self, tmp_path):
        """Test batch processing of multiple images."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        
        # Create test image files
        test_images = []
        for i in range(3):
            image_path = tmp_path / f"test_{i}.jpg"
            image_path.write_bytes(b"fake_image_data")
            test_images.append(image_path)
        
        identifier.start()
        
        results = identifier.process_batch(test_images)
        
        assert len(results) == 3
        assert all(isinstance(result, IdentificationResult) for result in results)
        assert all(result.species_name == "Unknown species" for result in results)
        
        identifier.stop()
    
    def test_species_identifier_error_handling(self):
        """Test error handling in species identifier."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        identifier.start()
        
        # Test with invalid image data
        with patch.object(identifier._implementation, 'identify_from_array') as mock_identify:
            mock_identify.side_effect = Exception("Processing error")
            
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = identifier.identify_from_array(test_image)
            
            # Should return fallback result on error
            assert result.species_name == "Unknown species"
            assert result.api_success is False
            assert "error" in result.fallback_reason.lower()
        
        identifier.stop()
    
    def test_species_identifier_timeout_handling(self):
        """Test timeout handling in species identifier."""
        config = Config.create_test_config()
        config.performance.api_timeout = 0.1  # Very short timeout
        
        identifier = SpeciesIdentifier(config, use_mock=True)
        identifier.start()
        
        # Mock long processing time
        with patch.object(identifier._implementation, 'identify_from_array') as mock_identify:
            def slow_process(image):
                time.sleep(0.2)  # Longer than timeout
                return IdentificationResult("Test Species", 0.9, 0.2, True)
            
            mock_identify.side_effect = slow_process
            
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = identifier.identify_from_array(test_image)
            
            # Should return timeout fallback
            assert result.species_name == "Unknown species"
            assert result.api_success is False
            assert "timeout" in result.fallback_reason.lower()
        
        identifier.stop()
    
    def test_species_identifier_confidence_filtering(self):
        """Test confidence threshold filtering."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        identifier.start()
        
        # Mock low confidence result
        with patch.object(identifier._implementation, 'identify_from_array') as mock_identify:
            mock_identify.return_value = IdentificationResult(
                species_name="European Hedgehog",
                confidence=0.3,  # Below typical threshold
                processing_time=1.0,
                api_success=True
            )
            
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = identifier.identify_from_array(test_image, min_confidence=0.5)
            
            # Should be filtered out due to low confidence
            assert result.species_name == "Unknown species"
            assert result.api_success is False
            assert "confidence" in result.fallback_reason.lower()
        
        identifier.stop()
    
    def test_species_identifier_retry_logic(self):
        """Test retry logic on failures."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        identifier.start()
        
        call_count = 0
        
        def failing_then_success(image):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return IdentificationResult("European Hedgehog", 0.8, 1.0, True)
        
        with patch.object(identifier._implementation, 'identify_from_array') as mock_identify:
            mock_identify.side_effect = failing_then_success
            
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = identifier.identify_from_array(test_image, max_retries=2)
            
            # Should succeed on second attempt
            assert result.species_name == "European Hedgehog"
            assert result.api_success is True
            assert call_count == 2
        
        identifier.stop()
    
    def test_species_identifier_system_info(self):
        """Test system information retrieval."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        
        system_info = identifier.get_system_info()
        
        assert 'implementation_type' in system_info
        assert 'configuration' in system_info
        assert 'stats' in system_info
        assert 'capabilities' in system_info
        
        assert system_info['implementation_type'] == 'MockSpeciesIdentifier'
        assert 'api_timeout' in system_info['configuration']
    
    def test_species_identifier_health_check(self):
        """Test health check functionality."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        
        # Test when stopped
        health = identifier.health_check()
        assert health['status'] == 'stopped'
        assert health['is_healthy'] is False
        
        # Test when running
        identifier.start()
        health = identifier.health_check()
        assert health['status'] == 'running'
        assert health['is_healthy'] is True
        
        identifier.stop()
    
    def test_species_identifier_cache_behavior(self):
        """Test caching behavior for repeated requests."""
        config = Config.create_test_config()
        identifier = SpeciesIdentifier(config, use_mock=True)
        identifier.start()
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Enable caching for this test
        with patch.object(identifier._implementation, 'identify_from_array') as mock_identify:
            mock_identify.return_value = IdentificationResult(
                "Cached Species", 0.9, 1.0, True
            )
            
            # First call should hit the implementation
            result1 = identifier.identify_from_array(test_image, use_cache=True)
            
            # Second call with same image might use cache (depending on implementation)
            result2 = identifier.identify_from_array(test_image, use_cache=True)
            
            assert result1.species_name == "Cached Species"
            assert result2.species_name == "Cached Species"
        
        identifier.stop()


# Future API integration test template
class TestAPISpeciesIdentifier:
    """Test template for future API implementations."""
    
    @pytest.mark.skip(reason="API implementation not yet available")
    def test_api_species_identifier_placeholder(self):
        """Placeholder test for future API implementation."""
        # This test will be implemented when real API integration is added
        # Should test:
        # - API authentication
        # - Network error handling
        # - Rate limiting
        # - Response parsing
        # - Real species identification
        pass


if __name__ == '__main__':
    pytest.main([__file__])