#!/usr/bin/env python3
"""
Test script to run the squirrel test image through the full detection and classification pipeline.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config
from species_identifier import SpeciesIdentifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the squirrel test image through the classification pipeline."""

    logger.info("=== Squirrel Test Image Classification ===")

    # Path to test image
    test_image_path = Path(__file__).parent.parent / 'data' / 'images' / 'squirrel_test_2.png'
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return 1
    
    logger.info(f"Test image: {test_image_path}")

    # Load config
    config = Config()

    try:
        # Initialize species identifier
        logger.info("\nInitializing SpeciesNet (this may take 10-15 seconds)...")
        identifier = SpeciesIdentifier(config)

        # Run identification
        logger.info("\nRunning species identification pipeline...")
        logger.info("Stage 1: Running MegaDetector to find animals...")
        result = identifier.identify_species(str(test_image_path))

        # Display results
        logger.info("\n" + "="*60)
        logger.info("RESULTS")
        logger.info("="*60)
        logger.info(f"Species: {result.species_name}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"API Success: {result.api_success}")
        logger.info(f"Processing Time: {result.processing_time:.2f}s")

        if result.fallback_reason:
            logger.info(f"Fallback Reason: {result.fallback_reason}")

        # Detection stage info
        if result.detection_result:
            logger.info(f"\n{'='*60}")
            logger.info("DETECTION STAGE")
            logger.info("="*60)
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
            logger.info(f"\n{'='*60}")
            logger.info("CLASSIFICATION METADATA")
            logger.info("="*60)
            if 'top_predictions' in result.metadata:
                logger.info("Top 5 predictions:")
                for i, pred in enumerate(result.metadata['top_predictions'][:5], 1):
                    logger.info(f"  {i}. {pred.get('label', 'unknown')}: "
                              f"{pred.get('score', 0):.2%}")

        logger.info(f"\n{'='*60}")
        logger.info("Test completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
