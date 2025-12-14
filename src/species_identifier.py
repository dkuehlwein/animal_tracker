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
class DetectionResult:
    """Result of MegaDetector animal detection."""
    animals_detected: bool
    detection_count: int
    bounding_boxes: list  # List of dicts with bbox coords and confidence
    detections: list  # Full detection info (category, conf, bbox)
    processing_time: float


@dataclass
class IdentificationResult:
    """Result of species identification."""
    species_name: str
    confidence: float
    api_success: bool
    processing_time: float
    fallback_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Two-stage pipeline info
    detection_result: Optional[DetectionResult] = None
    animals_detected: bool = True  # For backward compatibility


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
            from speciesnet import SpeciesNet

            # Build model name from config version (e.g., "kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
            # Default to v4.0.1a if not specified (to match downloaded model)
            model_version = self.config.species.model_version or "v4.0.1a"
            if not model_version.startswith("kaggle:") and not model_version.startswith("hf:"):
                # Construct full Kaggle model path
                model_name = f"kaggle:google/speciesnet/pyTorch/{model_version}/1"
            else:
                model_name = model_version

            # Initialize SpeciesNet with all components (detector + classifier + ensemble)
            # Geographic filtering is handled via predict() method parameters
            self._model = SpeciesNet(
                model_name=model_name,
                components='all',  # Load detector, classifier, and ensemble
                geofence=True  # Enable geographic filtering
            )

            self._model_loaded = True
            logger.info(f"SpeciesNet loaded successfully: {model_name} "
                       f"(Geofencing enabled for: {self.config.species.country_code}, "
                       f"Region: {self.config.species.admin1_region})")

        except Exception as e:
            logger.error(f"Failed to load SpeciesNet: {e}")
            raise SpeciesIdentificationError(f"Model initialization failed: {e}")

    def detect_animals(self, image_path, timeout=None) -> DetectionResult:
        """
        Stage 1: Use MegaDetector to find animals in image.

        Args:
            image_path: Path to image file
            timeout: Processing timeout (uses config default if None)

        Returns:
            DetectionResult with bounding boxes and detection info
        """
        start_time = time.time()

        try:
            # Validate image exists
            if not Path(image_path).exists():
                logger.error(f"Image not found: {image_path}")
                return DetectionResult(
                    animals_detected=False,
                    detection_count=0,
                    bounding_boxes=[],
                    detections=[],
                    processing_time=time.time() - start_time
                )

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Run MegaDetector only (don't classify yet)
            predictions = self._run_detection(image_path, timeout)

            # Parse detection results
            detection_result = self._parse_detections(predictions, start_time)

            return detection_result

        except Exception as e:
            logger.error(f"Animal detection failed: {e}")
            return DetectionResult(
                animals_detected=False,
                detection_count=0,
                bounding_boxes=[],
                detections=[],
                processing_time=time.time() - start_time
            )

    def classify_species(self, image_path, detection_result: DetectionResult, timeout=None) -> IdentificationResult:
        """
        Stage 2: Classify species from detected animal regions.

        Args:
            image_path: Path to image file
            detection_result: Result from detect_animals()
            timeout: Processing timeout (uses config default if None)

        Returns:
            IdentificationResult with species name, confidence, etc.
        """
        start_time = time.time()

        try:
            # Check if animals were detected
            if not detection_result.animals_detected:
                return IdentificationResult(
                    species_name='Unknown species',
                    confidence=0.0,
                    api_success=True,
                    processing_time=time.time() - start_time,
                    fallback_reason='No animals detected by MegaDetector',
                    detection_result=detection_result,
                    animals_detected=False
                )

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Run classification on detected regions
            predictions = self._run_classification(image_path, detection_result, timeout)

            # Parse classification results
            result = self._parse_classifications(predictions, start_time, detection_result)

            return result

        except Exception as e:
            logger.error(f"Species classification failed: {e}")
            return self._create_error_response(
                start_time, f"Classification error: {e}",
                detection_result=detection_result
            )

    def identify_species(self, image_path, timeout=None) -> IdentificationResult:
        """
        Full two-stage pipeline: Detect animals → Classify species.

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

            # Stage 1: Detect animals
            detection_result = self.detect_animals(image_path, timeout)

            # Stage 2: Classify species (only if animals detected)
            if detection_result.animals_detected:
                return self.classify_species(image_path, detection_result, timeout)
            else:
                # No animals detected - skip classification
                logger.info(f"No animals detected in {image_path}, skipping classification")
                return IdentificationResult(
                    species_name='Unknown species',
                    confidence=0.0,
                    api_success=True,
                    processing_time=time.time() - start_time,
                    fallback_reason='No animals detected by MegaDetector',
                    detection_result=detection_result,
                    animals_detected=False
                )

        except SpeciesIdentificationError as e:
            # Handle model loading failures gracefully
            logger.error(f"Species identification failed: {e}")
            return self._create_error_response(
                start_time, str(e)
            )
        except Exception as e:
            logger.error(f"Species identification failed: {e}")
            return self._create_error_response(
                start_time, f"Identification error: {e}"
            )

    def _run_detection(self, image_path: Path, timeout: float):
        """Run MegaDetector to find animals (Stage 1)."""
        try:
            # Run full prediction pipeline (detection + classification + ensemble)
            # SpeciesNet.predict() runs all stages and returns combined results
            predictions = self._model.predict(
                filepaths=[str(image_path)],
                country=self.config.species.country_code,
                admin1_region=self.config.species.admin1_region,
                run_mode='single_thread',  # Single image, no need for threading
                progress_bars=False
            )

            return predictions

        except Exception as e:
            logger.error(f"MegaDetector inference failed: {e}")
            raise SpeciesIdentificationError(f"Detection error: {e}")

    def _run_classification(self, image_path: Path, detection_result: DetectionResult, timeout: float):
        """Run SpeciesNet classifier on detected regions (Stage 2)."""
        try:
            # Note: In the full pipeline, predict() already runs classification
            # This method is kept for compatibility but just re-runs the full pipeline
            predictions = self._model.predict(
                filepaths=[str(image_path)],
                country=self.config.species.country_code,
                admin1_region=self.config.species.admin1_region,
                run_mode='single_thread',
                progress_bars=False
            )
            return predictions

        except Exception as e:
            logger.error(f"SpeciesNet classification failed: {e}")
            raise SpeciesIdentificationError(f"Classification error: {e}")

    def _run_inference(self, image_path: Path, timeout: float):
        """Legacy method - Run SpeciesNet inference with timeout protection."""
        return self._run_detection(image_path, timeout)

    def _parse_detections(self, predictions, start_time) -> DetectionResult:
        """
        Parse MegaDetector results from SpeciesNet predictions (Stage 1).

        Extracts detection information: bounding boxes, categories, confidence.
        """
        processing_time = time.time() - start_time

        if not predictions or not predictions.get('predictions'):
            return DetectionResult(
                animals_detected=False,
                detection_count=0,
                bounding_boxes=[],
                detections=[],
                processing_time=processing_time
            )

        pred = predictions['predictions'][0]
        detections = pred.get('detections', [])

        # Filter for animal detections above confidence threshold
        animal_detections = [
            d for d in detections
            if d['category'] == 'animal'
            and d['conf'] >= self.config.species.min_detection_confidence
        ]

        # Extract bounding boxes
        bounding_boxes = [
            {
                'bbox': d['bbox'],  # [x1, y1, x2, y2] normalized 0-1
                'confidence': d['conf'],
                'category': d['category']
            }
            for d in animal_detections
        ]

        animals_detected = len(animal_detections) > 0

        logger.info(f"MegaDetector found {len(animal_detections)} animals "
                   f"(total detections: {len(detections)}, processing: {processing_time:.2f}s)")

        return DetectionResult(
            animals_detected=animals_detected,
            detection_count=len(animal_detections),
            bounding_boxes=bounding_boxes,
            detections=detections,  # Keep full detection info
            processing_time=processing_time
        )

    def _parse_classifications(self, predictions, start_time, detection_result: DetectionResult) -> IdentificationResult:
        """
        Parse SpeciesNet classification results (Stage 2).

        Extracts species name and confidence from SpeciesNet predictions.
        """
        processing_time = time.time() - start_time

        if not predictions or not predictions.get('predictions'):
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=processing_time,
                fallback_reason='No classification results returned',
                detection_result=detection_result,
                animals_detected=detection_result.animals_detected
            )

        pred = predictions['predictions'][0]

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
                },
                detection_result=detection_result,
                animals_detected=detection_result.animals_detected
            )

        # Success case
        logger.info(f"Classified: {final_species} (confidence: {final_confidence:.2f})")
        return IdentificationResult(
            species_name=final_species,
            confidence=final_confidence,
            api_success=True,
            processing_time=processing_time,
            metadata={
                'top_predictions': pred.get('classifications', [])[:self.config.species.return_top_k]
            },
            detection_result=detection_result,
            animals_detected=detection_result.animals_detected
        )

    def _parse_predictions(self, predictions, start_time) -> IdentificationResult:
        """
        Legacy method: Parse SpeciesNet predictions using two-stage approach.
        Now delegates to _parse_detections and _parse_classifications.
        """
        # First parse detections
        detection_result = self._parse_detections(predictions, start_time)

        # Then parse classifications if animals were detected
        if detection_result.animals_detected:
            return self._parse_classifications(predictions, start_time, detection_result)
        else:
            # No animals detected
            all_categories = [d.get('category', 'unknown') for d in detection_result.detections]
            categories_str = ", ".join(set(all_categories)) if all_categories else "nothing"

            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=True,  # Model ran successfully
                processing_time=time.time() - start_time,
                fallback_reason=f'No animals detected (found: {categories_str})',
                metadata={'detections': detection_result.detections},
                detection_result=detection_result,
                animals_detected=False
            )

    def _create_error_response(self, start_time, reason, detection_result=None):
        """Create standardized error response."""
        return IdentificationResult(
            species_name='Unknown species',
            confidence=0.0,
            api_success=False,
            processing_time=time.time() - start_time,
            fallback_reason=reason,
            detection_result=detection_result,
            animals_detected=detection_result.animals_detected if detection_result else False
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

    def detect_animals(self, image_path, timeout=None) -> DetectionResult:
        """Mock animal detection (Stage 1)."""
        import random

        self.call_count += 1
        start_time = time.time()

        # Validate image exists
        if not Path(image_path).exists():
            return DetectionResult(
                animals_detected=False,
                detection_count=0,
                bounding_boxes=[],
                detections=[],
                processing_time=time.time() - start_time
            )

        # Simulate processing delay (0.05-0.15 seconds for detection)
        processing_delay = random.uniform(0.05, 0.15)
        time.sleep(processing_delay)

        # 80% chance of detecting an animal (realistic for motion-triggered cameras)
        if random.random() < 0.8:
            # Generate 1-3 mock detections
            num_detections = random.randint(1, 3)
            bounding_boxes = []
            detections = []

            for i in range(num_detections):
                # Generate random bbox (normalized 0-1)
                x1 = random.uniform(0.1, 0.5)
                y1 = random.uniform(0.1, 0.5)
                width = random.uniform(0.2, 0.4)
                height = random.uniform(0.2, 0.4)
                x2 = min(x1 + width, 0.9)
                y2 = min(y1 + height, 0.9)

                conf = random.uniform(0.7, 0.95)

                bbox_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'category': 'animal'
                }
                bounding_boxes.append(bbox_dict)

                detection_dict = {
                    'category': 'animal',
                    'conf': conf,
                    'bbox': [x1, y1, x2, y2]
                }
                detections.append(detection_dict)

            return DetectionResult(
                animals_detected=True,
                detection_count=num_detections,
                bounding_boxes=bounding_boxes,
                detections=detections,
                processing_time=time.time() - start_time
            )
        else:
            # No animals detected
            return DetectionResult(
                animals_detected=False,
                detection_count=0,
                bounding_boxes=[],
                detections=[],
                processing_time=time.time() - start_time
            )

    def classify_species(self, image_path, detection_result: DetectionResult, timeout=None) -> IdentificationResult:
        """Mock species classification (Stage 2)."""
        import random

        start_time = time.time()

        # Check if animals were detected
        if not detection_result.animals_detected:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=time.time() - start_time,
                fallback_reason='No animals detected',
                detection_result=detection_result,
                animals_detected=False
            )

        # Simulate processing delay (0.1-0.3 seconds for classification)
        processing_delay = random.uniform(0.1, 0.3)
        time.sleep(processing_delay)

        # 50% chance of identifying a species (more realistic with detected animals)
        if random.random() < 0.5:
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
                fallback_reason='Mock implementation',
                detection_result=detection_result,
                animals_detected=True
            )
        else:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=time.time() - start_time,
                fallback_reason='Mock implementation - low confidence',
                detection_result=detection_result,
                animals_detected=True
            )

    def identify_species(self, image_path, timeout=None) -> IdentificationResult:
        """Mock two-stage identification pipeline."""
        import random

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

        # Stage 1: Detect animals
        detection_result = self.detect_animals(image_path, timeout)

        # Stage 2: Classify species (only if animals detected)
        if detection_result.animals_detected:
            return self.classify_species(image_path, detection_result, timeout)
        else:
            # No animals detected - return early
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=time.time() - start_time,
                fallback_reason='Mock implementation - no animals detected',
                detection_result=detection_result,
                animals_detected=False
            )

    def reset_stats(self):
        """Reset call counter for testing."""
        self.call_count = 0
