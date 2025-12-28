#!/usr/bin/env python3
"""
Test script for Telegram bot integration.
Verifies bot connectivity, message sending, and photo capabilities.
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from notification_service import NotificationService
from utils import MotionVisualizer
from models import MotionResult
import cv2
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(output_path: Path) -> Path:
    """Create a test image for photo sending test."""
    # Create a simple test image with text
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 100, 50)  # Dark green background

    # Add text
    cv2.putText(img, "Telegram Bot Test", (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), img)
    return output_path


def create_test_image_with_motion(output_path: Path, config) -> tuple[Path, np.ndarray, MotionResult]:
    """Create a test image simulating a wildlife detection with fake motion data."""
    # Create a more realistic wildlife scene
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Create gradient background (sky to ground)
    for y in range(1080):
        color = int(30 + (y / 1080) * 70)
        img[y, :] = (color, color + 20, color)

    # Draw a simple "animal" (circle representing a squirrel or bird)
    animal_center = (960, 540)  # Center of image
    cv2.circle(img, animal_center, 80, (60, 100, 180), -1)  # Brown-ish animal
    cv2.circle(img, (940, 520), 20, (40, 40, 40), -1)  # Eye
    cv2.circle(img, (980, 520), 20, (40, 40, 40), -1)  # Eye

    # Add some "grass" or foliage
    for _ in range(50):
        x = np.random.randint(0, 1920)
        y = np.random.randint(800, 1080)
        size = np.random.randint(10, 40)
        cv2.circle(img, (x, y), size, (30, 80, 30), -1)

    # Add timestamp
    timestamp_text = f"Test Capture: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.putText(img, timestamp_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save the image
    cv2.imwrite(str(output_path), img)

    # Create fake motion frame (low-res grayscale)
    motion_frame = cv2.cvtColor(
        cv2.resize(img, config.camera.motion_detection_resolution),
        cv2.COLOR_BGR2GRAY
    )

    # Create fake motion result (simulating detection at center)
    motion_result = MotionResult(
        motion_detected=True,
        motion_area=4500,
        detection_confidence=1.0,
        center_x=320,  # Center of 640x480 frame
        center_y=240,
        contour_count=12
    )

    return output_path, motion_frame, motion_result


async def test_notification_service():
    """Run comprehensive notification service tests."""

    print("=" * 60)
    print("NOTIFICATION SERVICE TEST")
    print("=" * 60)
    print()

    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = Config()
        print(f"✓ Bot token configured: {config.telegram_token[:10]}...")
        print(f"✓ Chat ID: {config.telegram_chat_id}")
        print()

        # Initialize notification service
        print("🤖 Initializing notification service...")
        notification_service = NotificationService(config)
        print("✓ Notification service initialized")
        print()

        # Test 1: Get bot info
        print("Test 1: Bot Information")
        print("-" * 40)
        try:
            bot_info = await notification_service.bot.get_me()
            print(f"✓ Bot connected successfully!")
            print(f"  Bot name: @{bot_info.username}")
            print(f"  Bot ID: {bot_info.id}")
            print(f"  First name: {bot_info.first_name}")
            print()
        except Exception as e:
            print(f"✗ Failed to get bot info: {e}")
            return False
        
        # Test 2: Send simple text message
        print("Test 2: Send Text Message")
        print("-" * 40)
        try:
            test_message = f"🧪 Test message from Wildlife Detector\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            await notification_service.bot.send_message(
                chat_id=config.telegram_chat_id,
                text=test_message
            )
            print("✓ Text message sent successfully")
            print()
        except Exception as e:
            print(f"✗ Failed to send text message: {e}")
            print()
        
        # Test 3: Send detection notification
        print("Test 3: Send Detection Notification")
        print("-" * 40)
        try:
            species_result = {
                'species_name': 'European Robin',
                'confidence': 0.87,
                'scientific_name': 'Erithacus rubecula'
            }
            success = await notification_service.send_detection_notification(
                species_result=species_result,
                motion_area=3500,
                timestamp=datetime.now()
            )
            if success:
                print("✓ Detection notification sent successfully")
            else:
                print("✗ Detection notification failed")
            print()
        except Exception as e:
            print(f"✗ Failed to send detection notification: {e}")
            print()
        
        # Test 4: Send photo with caption
        print("Test 4: Send Photo with Caption")
        print("-" * 40)
        try:
            # Create test image
            test_image_path = config.storage.image_dir / "telegram_test.jpg"
            create_test_image(test_image_path)
            print(f"  Created test image: {test_image_path}")
            
            # Send photo
            caption = "📸 Test photo from Wildlife Detector"
            success = await notification_service.send_photo_with_caption(
                image_path=test_image_path,
                caption=caption
            )
            if success:
                print("✓ Photo sent successfully")
            else:
                print("✗ Photo sending failed")
            print()
        except Exception as e:
            print(f"✗ Failed to send photo: {e}")
            print()
        
        # Test 5: Send system status
        print("Test 5: Send System Status")
        print("-" * 40)
        try:
            success = await notification_service.send_system_status(
                memory_percent=45.2,
                storage_percent=62.8,
                image_count=42
            )
            if success:
                print("✓ System status sent successfully")
            else:
                print("✗ System status failed")
            print()
        except Exception as e:
            print(f"✗ Failed to send system status: {e}")
            print()

        # Test 6: Motion Visualization (NEW!)
        print("Test 6: Motion Visualization with Media Group")
        print("-" * 40)
        try:
            # Create test image with fake motion detection
            test_image_path = config.storage.image_dir / "motion_test_original.jpg"
            image_path, motion_frame, motion_result = create_test_image_with_motion(
                test_image_path, config
            )
            print(f"  Created test image: {test_image_path}")
            print(f"  Fake motion area: {motion_result.motion_area} px")
            print(f"  Motion center: ({motion_result.center_x}, {motion_result.center_y})")

            # Create annotated version
            annotated_path = MotionVisualizer.create_annotated_image(
                image_path, motion_frame, config, motion_result
            )

            if annotated_path:
                print(f"  Created annotated image: {annotated_path}")

                # Send both images as media group
                caption = (
                    "🎯 Motion Visualization Test\n"
                    f"Motion Area: {motion_result.motion_area} px\n"
                    f"Detection at center of frame"
                )
                success = await notification_service.send_media_group(
                    [image_path, annotated_path],
                    caption
                )

                if success:
                    print("✓ Media group sent successfully")
                    print("  You should see:")
                    print("    1. Original image with fake animal")
                    print("    2. Annotated image with motion markers")
                else:
                    print("✗ Media group sending failed")
            else:
                print("✗ Failed to create annotated image")

            print()
        except Exception as e:
            print(f"✗ Failed motion visualization test: {e}")
            logger.exception("Motion visualization test failed")
            print()

        # Summary
        print("=" * 60)
        print("✅ TELEGRAM BOT TEST COMPLETED")
        print("=" * 60)
        print()
        print("Check your Telegram chat to verify all messages arrived!")
        print()
        print("Expected in Telegram:")
        print("  - Text message")
        print("  - Detection notification (European Robin)")
        print("  - Single test photo")
        print("  - System status message")
        print("  - Media group with original + annotated images (Test 6)")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        logger.exception("Test failed with exception")
        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(test_notification_service())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
