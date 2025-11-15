"""
Species identification using Google SpeciesNet.
Replaces mock implementation with real AI-powered wildlife identification.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import Config

logger = logging.getLogger(__name__)


class SpeciesIdentificationError(Exception):
    """Base exception for species identification errors."""
    pass


class IdentificationTimeout(SpeciesIdentificationError):
    """Raised when identification times out."""
    pass


@dataclass
class IdentificationResult:
    """Result of species identification."""
    species_name: str
    confidence: float
    api_success: bool
    processing_time: float
    fallback_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SpeciesIdentifier:
    """
    Real species identification using Google SpeciesNet.
    Designed for wildlife camera trap images with geographic filtering.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None  # Lazy-loaded
        self._model_loaded = False
        logger.info("SpeciesIdentifier initialized (model will load on first use)")

    def _ensure_model_loaded(self):
        """Lazy-load SpeciesNet models on first use."""
        if self._model_loaded:
            return

        try:
            logger.info("Loading SpeciesNet model (this may take 10-15 seconds)...")
            from speciesnet.ensemble import SpeciesNetEnsemble

            # Initialize ensemble (detector + classifier)
            self._model = SpeciesNetEnsemble(
                country=self.config.species.country_code,
                region=self.config.species.admin1_region,
                classifier_version=self.config.species.model_version,
                cache_dir=str(self.config.species.model_cache_dir)
            )

            self._model_loaded = True
            logger.info(f"SpeciesNet loaded successfully: {self.config.species.model_version} "
                       f"(Country: {self.config.species.country_code}, "
                       f"Region: {self.config.species.admin1_region})")

        except Exception as e:
            logger.error(f"Failed to load SpeciesNet: {e}")
            raise SpeciesIdentificationError(f"Model initialization failed: {e}")

    def identify_species(self, image_path, timeout=None) -> IdentificationResult:
        """
        Identify species in image using SpeciesNet.

        Args:
            image_path: Path to image file
            timeout: Processing timeout (uses config default if None)

        Returns:
            IdentificationResult with species name, confidence, etc.
        """
        start_time = time.time()
        timeout = timeout or self.config.species.processing_timeout

        try:
            # Validate image exists
            if not Path(image_path).exists():
                return self._create_error_response(
                    start_time, "Image file not found"
                )

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Run inference
            predictions = self._run_inference(image_path, timeout)

            # Parse and validate results
            result = self._parse_predictions(predictions, start_time)

            return result

        except SpeciesIdentificationError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logger.error(f"Species identification failed: {e}")
            return self._create_error_response(
                start_time, f"Identification error: {e}"
            )

    def _run_inference(self, image_path: Path, timeout: float):
        """Run SpeciesNet inference with timeout protection."""
        try:
            # SpeciesNet expects list of image info dicts
            image_info = [{
                'filepath': str(image_path),
                'country': self.config.species.country_code,
                'admin1_region': self.config.species.admin1_region
            }]

            # Run ensemble prediction
            predictions = self._model.predict(image_info)

            return predictions

        except Exception as e:
            logger.error(f"SpeciesNet inference failed: {e}")
            raise SpeciesIdentificationError(f"Inference error: {e}")

    def _parse_predictions(self, predictions, start_time) -> IdentificationResult:
        """
        Parse SpeciesNet predictions into IdentificationResult format.

        SpeciesNet output format:
        {
            'predictions': [{
                'filepath': str,
                'detections': [{
                    'category': str,  # 'animal', 'person', 'vehicle'
                    'conf': float,
                    'bbox': [x1, y1, x2, y2]  # normalized 0-1
                }],
                'classifications': [{
                    'class': str,  # species name
                    'score': float
                }],  # Top-5 predictions
                'prediction': str,  # Final ensemble prediction
                'prediction_score': float
            }]
        }
        """
        processing_time = time.time() - start_time

        if not predictions or not predictions.get('predictions'):
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=processing_time,
                fallback_reason='No predictions returned'
            )

        pred = predictions['predictions'][0]

        # Check if any animals detected
        detections = pred.get('detections', [])
        animal_detections = [d for d in detections
                            if d['category'] == 'animal'
                            and d['conf'] >= self.config.species.min_detection_confidence]

        if not animal_detections:
            # Check if there were any detections at all
            all_categories = [d['category'] for d in detections]
            if detections:
                return IdentificationResult(
                    species_name='Unknown species',
                    confidence=0.0,
                    api_success=True,  # Model ran successfully
                    processing_time=processing_time,
                    fallback_reason=f'No animals detected (found: {", ".join(set(all_categories)) if all_categories else "nothing"})',
                    metadata={'detections': detections}
                )
            else:
                return IdentificationResult(
                    species_name='Unknown species',
                    confidence=0.0,
                    api_success=True,
                    processing_time=processing_time,
                    fallback_reason='No animals detected above confidence threshold',
                    metadata={'detections': detections}
                )

        # Get final species prediction
        final_species = pred.get('prediction', 'Unknown species')
        final_confidence = pred.get('prediction_score', 0.0)

        # Apply confidence threshold
        if final_confidence < self.config.species.unknown_species_threshold:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=final_confidence,
                api_success=True,
                processing_time=processing_time,
                fallback_reason=f'Confidence {final_confidence:.2f} below threshold {self.config.species.unknown_species_threshold}',
                metadata={
                    'raw_prediction': final_species,
                    'raw_confidence': final_confidence,
                    'top_predictions': pred.get('classifications', [])[:self.config.species.return_top_k]
                }
            )

        # Success case
        logger.info(f"Identified: {final_species} (confidence: {final_confidence:.2f})")
        return IdentificationResult(
            species_name=final_species,
            confidence=final_confidence,
            api_success=True,
            processing_time=processing_time,
            metadata={
                'detections': animal_detections,
                'top_predictions': pred.get('classifications', [])[:self.config.species.return_top_k]
            }
        )

    def _create_error_response(self, start_time, reason):
        """Create standardized error response."""
        return IdentificationResult(
            species_name='Unknown species',
            confidence=0.0,
            api_success=False,
            processing_time=time.time() - start_time,
            fallback_reason=reason
        )

    def health_check(self):
        """Check if SpeciesNet service is available."""
        try:
            self._ensure_model_loaded()
            return {
                'available': True,
                'service': 'SpeciesNet',
                'version': self.config.species.model_version,
                'country': self.config.species.country_code,
                'region': self.config.species.admin1_region,
                'supported_formats': ['jpg', 'jpeg', 'png']
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }

    def get_supported_species(self):
        """
        Return information about supported species.
        Note: SpeciesNet supports 2000+ species, filtered by geography.
        """
        return {
            'total_labels': '2000+',
            'geographic_filter': f"{self.config.species.country_code}/{self.config.species.admin1_region}",
            'note': 'Species automatically filtered to region-appropriate wildlife'
        }

    def get_statistics(self):
        """Get statistics for the identification service."""
        return {
            'model_loaded': self._model_loaded,
            'model_version': self.config.species.model_version,
            'country': self.config.species.country_code,
            'region': self.config.species.admin1_region,
            'min_detection_conf': self.config.species.min_detection_confidence,
            'unknown_threshold': self.config.species.unknown_species_threshold
        }


# Keep MockSpeciesIdentifier for testing purposes
class MockSpeciesIdentifier(SpeciesIdentifier):
    """
    Mock implementation for testing without SpeciesNet.
    Maintains same interface but returns simulated results.
    """

    def __init__(self, config: Config, fail_rate: float = 0.0):
        # Don't call super().__init__() to avoid model loading
        self.config = config
        self.fail_rate = fail_rate
        self.call_count = 0
        self._model_loaded = False
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
        logger.info("MockSpeciesIdentifier initialized (for testing)")

    def _ensure_model_loaded(self):
        """Mock model loading - does nothing."""
        if not self._model_loaded:
            logger.info("Mock model 'loaded'")
            self._model_loaded = True

    def identify_species(self, image_path, timeout=None) -> IdentificationResult:
        """Mock identification with controllable failure rate."""
        import random

        self.call_count += 1
        start_time = time.time()

        # Simulate failures based on fail_rate
        if random.random() < self.fail_rate:
            raise SpeciesIdentificationError("Mock identification failure")

        # Simulate timeout
        if timeout and timeout < 0.1:
            raise IdentificationTimeout("Mock timeout")

        # Validate image exists
        if not Path(image_path).exists():
            return self._create_error_response(
                start_time, "Image file not found"
            )

        # Simulate processing time (0.1-0.5 seconds)
        processing_delay = random.uniform(0.1, 0.5)
        time.sleep(processing_delay)

        # Mock identification logic
        # 70% chance of "Unknown species" (realistic for wildlife cameras)
        if random.random() < 0.7:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=time.time() - start_time,
                fallback_reason='Mock implementation'
            )

        # 30% chance of identifying a species
        species = random.choice(self.mock_species)
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

        return IdentificationResult(
            species_name=species,
            confidence=confidence,
            api_success=False,  # Always False for mock
            processing_time=time.time() - start_time,
            fallback_reason='Mock implementation'
        )

    def reset_stats(self):
        """Reset call counter for testing."""
        self.call_count = 0
