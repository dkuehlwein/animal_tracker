"""
Notification service for the wildlife detection system.

Consolidates Telegram bot functionality and message formatting
into a unified notification interface.
"""

import logging
import telegram
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from config import Config

logger = logging.getLogger(__name__)


class NotificationFormatter:
    """Message formatting utilities for notifications."""

    @staticmethod
    def format_detection_message(species_name: str, confidence: float,
                                 motion_area: int, timestamp: datetime,
                                 temperature: Optional[float] = None) -> str:
        """Format detection message for Telegram."""
        time_str = timestamp.strftime("%H:%M")

        if species_name == "Unknown species":
            return f"🔍 Unknown species detected at {time_str}\nMotion area: {motion_area:,} pixels"

        # Emoji mapping
        emoji_map = {
            "Hedgehog": "🦔", "Fox": "🦊", "Squirrel": "🐿️",
            "Cat": "🐱", "Bird": "🐦", "Robin": "🐦", "Blackbird": "🐦"
        }

        emoji = "🔍"
        for animal, symbol in emoji_map.items():
            if animal in species_name:
                emoji = symbol
                break

        if confidence > 0.8:
            message = f"{emoji} {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}%"
        else:
            message = f"🔍 Possible {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}%"

        if temperature is not None:
            message += f"\n🌡️ {temperature:.1f}°C"

        return message

    @staticmethod
    def format_system_status(memory_percent: float, storage_percent: float,
                             image_count: int) -> str:
        """Format system status message."""
        return (f"📊 System Status\n"
                f"Memory: {memory_percent:.1f}% used\n"
                f"Storage: {storage_percent:.1f}% used\n"
                f"Images stored: {image_count}")


class NotificationService:
    """
    Unified notification service for wildlife detection system.

    Provides Telegram bot functionality with integrated message formatting.
    """

    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.telegram_token)
        self.formatter = NotificationFormatter()
        self._database = None

    def set_database_reference(self, database) -> None:
        """Set database reference for first sighting checks."""
        self._database = database

    async def send_detection_notification(self, species_result: dict,
                                          motion_area: int, timestamp: datetime,
                                          temperature: float = None) -> bool:
        """Send species detection notification to Telegram."""
        try:
            species_name = species_result.get('species_name', 'Unknown species')
            confidence = species_result.get('confidence', 0.0)

            # Use formatter to create message
            message = self.formatter.format_detection_message(
                species_name, confidence, motion_area, timestamp, temperature
            )

            # Check if this is first sighting today (if database is available)
            if self._database and species_name != "Unknown species":
                is_first_today = self._database.is_first_detection_today(species_name)
                if is_first_today:
                    message += " - First sighting today!"

            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message
            )

            return True

        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    async def send_photo_with_caption(self, image_path: Path,
                                      caption: str = None) -> bool:
        """Send photo to Telegram channel with optional caption."""
        try:
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return False

            default_caption = f"Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            caption = caption or default_caption

            with open(image_path, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=self.config.telegram_chat_id,
                    photo=photo,
                    caption=caption
                )

            return True

        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}")
            return False

    async def send_media_group(self, image_paths: List[Path], caption: str = None) -> bool:
        """Send multiple photos as a media group to Telegram channel."""
        files = []
        try:
            if not image_paths:
                logger.warning("No images provided to send_media_group")
                return False

            # Filter out non-existent files
            valid_paths = [p for p in image_paths if p.exists()]
            if not valid_paths:
                logger.warning("No valid image files found")
                return False

            # Create media group (max 10 images per Telegram limitation)
            # Open all files and keep handles in list
            media = []
            for i, path in enumerate(valid_paths[:10]):
                # Open file and keep handle in list
                file_handle = open(path, 'rb')
                files.append(file_handle)

                # Only first photo gets the caption
                if i == 0 and caption:
                    media.append(telegram.InputMediaPhoto(
                        media=file_handle,
                        caption=caption
                    ))
                else:
                    media.append(telegram.InputMediaPhoto(media=file_handle))

            await self.bot.send_media_group(
                chat_id=self.config.telegram_chat_id,
                media=media,
                read_timeout=30,
                write_timeout=30,
                connect_timeout=30
            )

            return True

        except telegram.error.TimedOut as e:
            # Telegram timeout errors are often false alarms - images may still be uploaded
            logger.warning(f"Telegram API timeout (images may still have been sent): {e}")
            # Return True anyway since images often go through despite timeout
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram media group: {e}")
            return False
        finally:
            # Close all file handles
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

    async def send_system_status(self, memory_percent: float,
                                 storage_percent: float, image_count: int) -> bool:
        """Send system status message."""
        try:
            message = self.formatter.format_system_status(
                memory_percent, storage_percent, image_count
            )

            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message
            )

            return True

        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False

    async def send_text_message(self, message: str) -> bool:
        """Send simple text message."""
        try:
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message
            )
            return True

        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Telegram bot connected: {bot_info.username}")
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
