"""
Telegram notification service for wildlife detection system.
Provides centralized Telegram bot functionality with message formatting.
"""

import asyncio
import logging
import telegram
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import Config
from utils import TelegramFormatter

logger = logging.getLogger(__name__)


class TelegramError(Exception):
    """Base exception for Telegram-related errors."""
    pass


class TelegramService:
    """Centralized Telegram notification service."""
    
    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.telegram_token)
        self.formatter = TelegramFormatter()
    
    async def send_detection_notification(self, species_result: dict, 
                                         motion_area: int, timestamp: datetime) -> bool:
        """Send species detection notification to Telegram."""
        try:
            species_name = species_result.get('species_name', 'Unknown species')
            confidence = species_result.get('confidence', 0.0)
            
            # Use TelegramFormatter to create message
            message = self.formatter.format_detection_message(
                species_name, confidence, motion_area, timestamp
            )
            
            # Check if this is first sighting today (if database is available)
            if hasattr(self, 'database') and species_name != "Unknown species":
                is_first_today = self.database.is_first_detection_today(species_name)
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
    
    def set_database_reference(self, database):
        """Set database reference for first sighting checks."""
        self.database = database
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Telegram bot connected: {bot_info.username}")
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False