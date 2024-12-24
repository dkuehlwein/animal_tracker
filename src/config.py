from pathlib import Path
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Telegram settings
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.telegram_token or not self.telegram_chat_id:
            raise ValueError("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        
        # Camera settings
        self.main_resolution = (1920, 1080)
        self.lores_resolution = (160, 120)
        self.frame_format = "RGB888"
        self.frame_duration = 100000  # 10 FPS limit
        
        # Motion detection settings
        self.motion_threshold = 30
        self.min_contour_area = 50
        self.background_history = 100
        self.background_threshold = 20
        self.frame_interval = 0.2  # 5 FPS
        self.consecutive_detections_required = 2
        self.central_region_bounds = (0.2, 0.8)  # 20% from edges
        self.center_weight = 1.0
        self.edge_weight = 0.2
        
        # Storage settings
        self.max_images = 100
        self.data_dir = Path("data")
        self.image_prefix = "capture_"
        
        # Timing settings
        self.cooldown_period = 10  # seconds
        self.startup_delay = 2  # seconds
        self.idle_sleep = 0.05  # seconds
        self.cooldown_sleep = 0.1  # seconds
        self.error_sleep = 5  # seconds 