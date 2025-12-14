#!/usr/bin/env python3
"""
Manual test script for species classification pipeline.
Captures a photo and runs it through the SpeciesNet identification.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config
from camera_manager import CameraManager
from species_identifier import SpeciesIdentifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Capture a test photo and run species classification."""

    logger.info("=== Species Classification Test ===")

    # Load config
    config = Config()

    # Initialize camera
    logger.info("Initializing camera...")
    camera = CameraManager(config)

    try:
        # Start camera in context manager
        with camera.camera_session():
            # Capture a test photo
            logger.info("Capturing test photo...")
            test_image_path = camera.capture_and_save_photo()
            logger.info(f"Photo saved to: {test_image_path}")

            # Initialize species identifier
            logger.info("\nInitializing SpeciesNet (this may take 10-15 seconds)...")
            identifier = SpeciesIdentifier(config)

            # Run identification
            logger.info("\nRunning species identification...")
            logger.info("Stage 1: Running MegaDetector to find animals...")
            result = identifier.identify_species(test_image_path)

            # Display results
            logger.info("\n=== RESULTS ===")
            logger.info(f"Species: {result.species_name}")
            logger.info(f"Confidence: {result.confidence:.2%}")
            logger.info(f"API Success: {result.api_success}")
            logger.info(f"Processing Time: {result.processing_time:.2f}s")

            if result.fallback_reason:
                logger.info(f"Fallback Reason: {result.fallback_reason}")

            # Detection stage info
            if result.detection_result:
                logger.info(f"\n=== DETECTION STAGE ===")
                logger.info(f"Animals Detected: {result.detection_result.animals_detected}")
                logger.info(f"Detection Count: {result.detection_result.detection_count}")
                logger.info(f"Detection Time: {result.detection_result.processing_time:.2f}s")

                if result.detection_result.bounding_boxes:
                    logger.info(f"\nBounding Boxes:")
                    for i, bbox in enumerate(result.detection_result.bounding_boxes, 1):
                        logger.info(f"  {i}. Category: {bbox['category']}, "
                                  f"Confidence: {bbox['confidence']:.2%}, "
                                  f"BBox: {bbox['bbox']}")

            # Metadata
            if result.metadata:
                logger.info(f"\n=== METADATA ===")
                if 'top_predictions' in result.metadata:
                    logger.info("Top predictions:")
                    for pred in result.metadata['top_predictions'][:5]:
                        logger.info(f"  - {pred.get('label', 'unknown')}: "
                                  f"{pred.get('score', 0):.2%}")

            logger.info(f"\nTest image location: {test_image_path}")
            logger.info("You can view the image to verify the classification")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
