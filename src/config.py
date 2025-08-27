"""
Configuration management for Wildlife Detector system.
Provides type-safe configuration with validation and environment-based overrides.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """Camera-specific configuration settings."""
    main_resolution: Tuple[int, int] = (1920, 1080)
    motion_detection_resolution: Tuple[int, int] = (640, 480)
    motion_detection_format: str = "YUV420"
    api_capture_resolution: Tuple[int, int] = (1920, 1080)
    frame_format: str = "RGB888"
    frame_duration: int = 100000  # microseconds
    startup_delay: float = 2.0
    
    def __post_init__(self):
        """Validate camera configuration."""
        if not self._validate_resolution(self.main_resolution):
            raise ValueError(f"Invalid main resolution: {self.main_resolution}")
        if not self._validate_resolution(self.motion_detection_resolution):
            raise ValueError(f"Invalid motion detection resolution: {self.motion_detection_resolution}")
        if self.motion_detection_format not in ["YUV420", "RGB888"]:
            raise ValueError(f"Invalid motion detection format: {self.motion_detection_format}")
    
    @staticmethod
    def _validate_resolution(resolution: Tuple[int, int]) -> bool:
        """Validate resolution tuple."""
        return (len(resolution) == 2 and 
                all(isinstance(x, int) and x > 0 for x in resolution))


@dataclass(frozen=True)
class MotionConfig:
    """Motion detection configuration settings."""
    motion_threshold: int = 2000
    min_contour_area: int = 50
    background_history: int = 100
    background_threshold: int = 20
    frame_interval: float = 0.2  # 5 FPS
    consecutive_detections_required: int = 2
    central_region_bounds: Tuple[float, float] = (0.2, 0.8)
    center_weight: float = 1.0
    edge_weight: float = 0.2
    
    def __post_init__(self):
        """Validate motion detection configuration."""
        if self.motion_threshold <= 0:
            raise ValueError("Motion threshold must be positive")
        if not (0.0 < self.central_region_bounds[0] < self.central_region_bounds[1] < 1.0):
            raise ValueError("Invalid central region bounds")
        if self.center_weight <= 0 or self.edge_weight < 0:
            raise ValueError("Invalid weight values")


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance and resource management configuration."""
    cooldown_period: float = 30.0
    memory_threshold: float = 0.8
    processing_timeout: float = 30.0
    max_images: int = 100
    idle_sleep: float = 0.05
    cooldown_sleep: float = 0.1
    error_sleep: float = 5.0
    cleanup_days: int = 30
    
    def __post_init__(self):
        """Validate performance configuration."""
        if not (0.0 < self.memory_threshold < 1.0):
            raise ValueError("Memory threshold must be between 0 and 1")
        if self.max_images <= 0:
            raise ValueError("Max images must be positive")


@dataclass(frozen=True)
class StorageConfig:
    """Storage and file management configuration."""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    image_dir: Path = field(default_factory=lambda: Path("data/images"))
    logs_dir: Path = field(default_factory=lambda: Path("data/logs"))
    database_path: str = "data/detections.db"
    image_prefix: str = "capture_"
    
    def __post_init__(self):
        """Validate and ensure directories exist."""
        for directory in [self.data_dir, self.image_dir, self.logs_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """
    Main configuration class that aggregates all configuration sections.
    Provides environment variable overrides and validation.
    """
    
    def __init__(self, env_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file
            
        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        # Load environment variables
        load_dotenv(env_file)
        
        # Initialize sub-configurations
        self.camera = self._load_camera_config()
        self.motion = self._load_motion_config()
        self.performance = self._load_performance_config()
        self.storage = self._load_storage_config()
        
        # Load Telegram settings
        self.telegram_token = self._get_required_env("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = self._get_required_env("TELEGRAM_CHAT_ID")
        
        # Validate complete configuration
        self._validate_configuration()
        
        logger.info("Configuration loaded successfully")
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(f"Required environment variable {key} is not set")
        return value
    
    def _get_optional_env(self, key: str, default: str, converter=str):
        """Get optional environment variable with type conversion."""
        value = os.getenv(key, default)
        try:
            return converter(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid value for {key}: {value}, using default: {default}")
            return converter(default)
    
    def _load_camera_config(self) -> CameraConfig:
        """Load camera configuration with environment overrides."""
        return CameraConfig(
            main_resolution=self._parse_resolution(
                os.getenv("CAMERA_MAIN_RESOLUTION", "1920x1080")
            ),
            motion_detection_resolution=self._parse_resolution(
                os.getenv("CAMERA_MOTION_RESOLUTION", "640x480")
            ),
            motion_detection_format=self._get_optional_env(
                "CAMERA_MOTION_FORMAT", "YUV420"
            ),
            frame_duration=self._get_optional_env(
                "CAMERA_FRAME_DURATION", "100000", int
            ),
            startup_delay=self._get_optional_env(
                "CAMERA_STARTUP_DELAY", "2.0", float
            )
        )
    
    def _load_motion_config(self) -> MotionConfig:
        """Load motion detection configuration with environment overrides."""
        return MotionConfig(
            motion_threshold=self._get_optional_env(
                "MOTION_THRESHOLD", "2000", int
            ),
            consecutive_detections_required=self._get_optional_env(
                "MOTION_CONSECUTIVE_REQUIRED", "2", int
            ),
            frame_interval=self._get_optional_env(
                "MOTION_FRAME_INTERVAL", "0.2", float
            )
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration with environment overrides."""
        return PerformanceConfig(
            cooldown_period=self._get_optional_env(
                "PERFORMANCE_COOLDOWN", "30.0", float
            ),
            memory_threshold=self._get_optional_env(
                "PERFORMANCE_MEMORY_THRESHOLD", "0.8", float
            ),
            max_images=self._get_optional_env(
                "PERFORMANCE_MAX_IMAGES", "100", int
            )
        )
    
    def _load_storage_config(self) -> StorageConfig:
        """Load storage configuration with environment overrides."""
        data_dir = Path(self._get_optional_env("STORAGE_DATA_DIR", "data"))
        return StorageConfig(
            data_dir=data_dir,
            image_dir=data_dir / "images",
            logs_dir=data_dir / "logs",
            database_path=self._get_optional_env(
                "STORAGE_DATABASE_PATH", str(data_dir / "detections.db")
            )
        )
    
    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse resolution string like '1920x1080' to tuple."""
        try:
            width, height = resolution_str.split('x')
            return (int(width), int(height))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid resolution format: {resolution_str}, using default")
            return (1920, 1080)
    
    def _validate_configuration(self):
        """Validate complete configuration consistency."""
        # Check that motion detection resolution is smaller than main resolution
        if (self.camera.motion_detection_resolution[0] > self.camera.main_resolution[0] or
            self.camera.motion_detection_resolution[1] > self.camera.main_resolution[1]):
            raise ConfigurationError(
                "Motion detection resolution cannot be larger than main resolution"
            )
        
        # Validate database path directory exists
        db_path = Path(self.storage.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "camera": {
                "main_resolution": self.camera.main_resolution,
                "motion_resolution": self.camera.motion_detection_resolution,
                "motion_format": self.camera.motion_detection_format
            },
            "motion": {
                "threshold": self.motion.motion_threshold,
                "frame_interval": self.motion.frame_interval,
                "consecutive_required": self.motion.consecutive_detections_required
            },
            "performance": {
                "cooldown_period": self.performance.cooldown_period,
                "memory_threshold": self.performance.memory_threshold,
                "max_images": self.performance.max_images
            },
            "storage": {
                "data_dir": str(self.storage.data_dir),
                "database_path": self.storage.database_path
            }
        }
    
    @classmethod
    def create_test_config(cls, **overrides) -> 'Config':
        """Create configuration for testing with optional overrides."""
        # Set test environment variables
        test_env = {
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat',
        }
        test_env.update(overrides)
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.getenv(key)
            os.environ[key] = value
        
        try:
            config = cls()
            return config
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value