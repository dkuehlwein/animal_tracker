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
from telegram_service import TelegramService
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


async def test_telegram_service():
    """Run comprehensive Telegram service tests."""
    
    print("=" * 60)
    print("TELEGRAM BOT TEST")
    print("=" * 60)
    print()
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = Config()
        print(f"✓ Bot token configured: {config.telegram_token[:10]}...")
        print(f"✓ Chat ID: {config.telegram_chat_id}")
        print()
        
        # Initialize Telegram service
        print("🤖 Initializing Telegram service...")
        telegram_service = TelegramService(config)
        print("✓ Telegram service initialized")
        print()
        
        # Test 1: Get bot info
        print("Test 1: Bot Information")
        print("-" * 40)
        try:
            bot_info = await telegram_service.bot.get_me()
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
            await telegram_service.bot.send_message(
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
            success = await telegram_service.send_detection_notification(
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
            success = await telegram_service.send_photo_with_caption(
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
            success = await telegram_service.send_system_status(
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
        
        # Summary
        print("=" * 60)
        print("✅ TELEGRAM BOT TEST COMPLETED")
        print("=" * 60)
        print()
        print("Check your Telegram chat to verify all messages arrived!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        logger.exception("Test failed with exception")
        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(test_telegram_service())
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
