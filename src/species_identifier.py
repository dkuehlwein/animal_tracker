import time
import random
from pathlib import Path
from config import Config

class SpeciesIdentifier:
    """
    Mock species identification service for wildlife camera.
    Designed with interface that supports future integration of:
    - SpeciesNet cloud API
    - Google Vision API
    - Local lightweight models
    - Other identification services
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mock_species = [
            "European Hedgehog",
            "Red Fox", 
            "Grey Squirrel",
            "European Robin",
            "Blackbird",
            "Wood Pigeon",
            "Domestic Cat",
            "Unknown Bird",
            "Small Mammal"
        ]
    
    def identify_species(self, image_path, timeout=None):
        """
        Mock species identification that returns consistent format
        for future API integration.
        
        Args:
            image_path: Path to the captured image
            timeout: Processing timeout (simulated)
            
        Returns:
            dict: {
                'species_name': str,
                'confidence': float (0.0-1.0),
                'api_success': bool,
                'processing_time': float,
                'fallback_reason': str or None
            }
        """
        start_time = time.time()
        
        try:
            # Validate image exists
            if not Path(image_path).exists():
                return self._create_error_response(
                    start_time, 
                    "Image file not found"
                )
            
            # Simulate processing time (0.1-0.5 seconds)
            processing_delay = random.uniform(0.1, 0.5)
            time.sleep(processing_delay)
            
            # Mock identification logic
            # In real implementation, this would call external API or model
            result = self._mock_identification()
            
            processing_time = time.time() - start_time
            
            return {
                'species_name': result['species'],
                'confidence': result['confidence'],
                'api_success': False,  # Always False for mock
                'processing_time': processing_time,
                'fallback_reason': 'Mock implementation'
            }
            
        except Exception as e:
            return self._create_error_response(
                start_time,
                f"Mock identification error: {e}"
            )
    
    def _mock_identification(self):
        """
        Mock identification logic.
        Returns different species with varying confidence levels.
        """
        # 70% chance of "Unknown species" (realistic for wildlife cameras)
        if random.random() < 0.7:
            return {
                'species': 'Unknown species',
                'confidence': 0.0
            }
        
        # 30% chance of identifying a species
        species = random.choice(self.mock_species)
        
        # Confidence varies by species (some easier to identify)
        confidence_ranges = {
            "European Hedgehog": (0.6, 0.9),
            "Red Fox": (0.7, 0.95),
            "Grey Squirrel": (0.5, 0.8),
            "Domestic Cat": (0.8, 0.95),
            "European Robin": (0.4, 0.7),
            "Blackbird": (0.4, 0.7),
            "Wood Pigeon": (0.3, 0.6),
            "Unknown Bird": (0.2, 0.4),
            "Small Mammal": (0.2, 0.4)
        }
        
        min_conf, max_conf = confidence_ranges.get(species, (0.1, 0.3))
        confidence = random.uniform(min_conf, max_conf)
        
        return {
            'species': species,
            'confidence': confidence
        }
    
    def _create_error_response(self, start_time, reason):
        """Create standardized error response"""
        return {
            'species_name': 'Unknown species',
            'confidence': 0.0,
            'api_success': False,
            'processing_time': time.time() - start_time,
            'fallback_reason': reason
        }
    
    def health_check(self):
        """
        Check if the identification service is available.
        For mock implementation, always returns True.
        """
        return {
            'available': True,
            'service': 'Mock Species Identifier',
            'version': '1.0.0',
            'supported_formats': ['jpg', 'jpeg', 'png']
        }
    
    def get_supported_species(self):
        """Return list of species this service can potentially identify"""
        return self.mock_species.copy()
    
    def get_statistics(self):
        """Mock statistics for the identification service"""
        return {
            'total_identifications': random.randint(0, 100),
            'success_rate': 0.3,  # Mock 30% identification rate
            'average_confidence': 0.65,
            'most_common_species': "European Hedgehog",
            'service_uptime': "100%"  # Mock always available
        }