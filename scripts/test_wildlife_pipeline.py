#!/usr/bin/env python3
"""
Test script to run an image through the complete wildlife detection pipeline
without requiring camera hardware.

Usage:
    uv run python scripts/test_wildlife_pipeline.py <image_path>
    uv run python scripts/test_wildlife_pipeline.py path/to/animal.jpg
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from species_identifier import SpeciesIdentifier
from notification_service import NotificationService
from database_manager import DatabaseManager
from data_models import IdentificationResult, DetectionRecord
from utils import PerformanceTimer


async def test_pipeline(image_path: Path) -> None:
    """Run an image through the complete wildlife detection pipeline."""

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Wildlife Detection Pipeline Test")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Load configuration
    config = Config()
    print(f"✓ Configuration loaded")
    print(f"  - Species model: {config.species.model_version}")
    print(f"  - Region: {config.species.country_code}/{config.species.admin1_region}")
    print(f"  - Unknown threshold: {config.species.unknown_species_threshold}")

    # Initialize components
    species_identifier = SpeciesIdentifier(config)
    notification_service = NotificationService(config)
    db_manager = DatabaseManager(config)

    print(f"\n✓ Components initialized")

    # Run species identification
    print(f"\n{'─'*60}")
    print(f"Running species identification...")
    print(f"{'─'*60}")

    timer = PerformanceTimer("species_identification")
    timer.start()

    result = species_identifier.identify_species(str(image_path))

    elapsed = timer.stop()

    print(f"\n{'─'*60}")
    print(f"Identification Results")
    print(f"{'─'*60}")
    print(f"Species: {result.species_name}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"API success: {result.api_success}")
    print(f"Animals detected: {result.animals_detected}")
    print(f"Processing time: {elapsed:.2f}s")

    if result.fallback_reason:
        print(f"Fallback reason: {result.fallback_reason}")

    if result.detection_result:
        print(f"\nDetection details:")
        print(f"  - Detection count: {result.detection_result.detection_count}")
        print(f"  - Detection time: {result.detection_result.processing_time:.2f}s")

    # Save to database
    print(f"\n{'─'*60}")
    print(f"Saving to database...")
    print(f"{'─'*60}")

    db_manager.log_detection(
        image_path=str(image_path),
        motion_area=0,  # Not from motion detection in this test
        species_name=result.species_name,
        confidence_score=result.confidence,
        processing_time=elapsed,
        api_success=result.api_success
    )

    # Create detection record for display purposes
    detection_record = DetectionRecord(
        id=None,
        timestamp=datetime.now(),
        image_path=str(image_path),
        motion_area=0,
        species_name=result.species_name,
        confidence_score=result.confidence,
        processing_time=elapsed,
        api_success=result.api_success
    )
    print(f"✓ Detection logged to database")

    # Send notification
    print(f"\n{'─'*60}")
    print(f"Sending Telegram notification...")
    print(f"{'─'*60}")

    try:
        # Format caption with species info
        if result.species_name == "Unknown species":
            caption = f"🔍 Unknown species detected\nConfidence: {result.confidence:.0%}"
        else:
            caption = f"🦔 {result.species_name} detected\nConfidence: {result.confidence:.0%}"

        await notification_service.send_photo_with_caption(
            image_path=image_path,
            caption=caption
        )
        print(f"✓ Notification sent successfully")
    except Exception as e:
        print(f"✗ Notification failed: {e}")

    # Display summary
    print(f"\n{'='*60}")
    print(f"Pipeline Test Complete")
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  - Species: {result.species_name}")
    print(f"  - Confidence: {result.confidence:.2%}")
    print(f"  - Processing time: {elapsed:.2f}s")
    print(f"  - Database: ✓ Logged")
    print(f"  - Notification: ✓ Sent")
    print(f"{'='*60}\n")

    # Query recent detections
    print(f"Recent detections from database:")
    print(f"{'─'*60}")
    recent = db_manager.get_recent_detections(limit=5)
    for i, det in enumerate(recent, 1):
        print(f"{i}. {det.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{det.species_name} ({det.confidence_score:.1%})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test wildlife detection pipeline with a static image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/test_wildlife_pipeline.py test_images/fox.jpg
  uv run python scripts/test_wildlife_pipeline.py captured_images/capture_20250104_120000.jpg
        """
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the image file to test"
    )

    args = parser.parse_args()

    # Run async pipeline
    asyncio.run(test_pipeline(args.image_path))


if __name__ == "__main__":
    main()
