"""
Configuration management for Wildlife Detector system.
Uses pydantic-settings for type-safe configuration with environment variable overrides.
"""

from pathlib import Path
from typing import Tuple, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)


class CameraConfig(BaseSettings):
    """Camera-specific configuration settings."""
    model_config = SettingsConfigDict(env_prefix='CAMERA_')

    main_resolution: Tuple[int, int] = (1920, 1080)
    motion_detection_resolution: Tuple[int, int] = (640, 480)
    motion_detection_format: str = "YUV420"
    frame_format: str = "RGB888"  # Main stream format for capture
    frame_duration: int = 100000  # microseconds
    startup_delay: float = 2.0
    exposure_time: Optional[int] = None  # None = auto-exposure
    analogue_gain: Optional[float] = None  # None = auto-gain

    @field_validator('motion_detection_format')
    @classmethod
    def validate_format(cls, v):
        if v not in ["YUV420", "RGB888"]:
            raise ValueError(f"Invalid motion detection format: {v}")
        return v

    @field_validator('exposure_time')
    @classmethod
    def validate_exposure(cls, v):
        if v is not None and (v < 100 or v > 1000000):
            raise ValueError(f"Exposure time must be between 100-1000000 μs")
        return v

    @field_validator('analogue_gain')
    @classmethod
    def validate_gain(cls, v):
        if v is not None and (v < 1.0 or v > 8.0):
            raise ValueError(f"Analogue gain must be between 1.0-8.0")
        return v


class MotionConfig(BaseSettings):
    """Motion detection configuration settings."""
    model_config = SettingsConfigDict(env_prefix='MOTION_')

    threshold: int = 2000
    min_contour_area: int = 50
    background_history: int = 100
    background_threshold: int = 40
    frame_interval: float = 0.2  # 5 FPS
    consecutive_required: int = 2
    blur_kernel_size: int = 5
    central_region_bounds: Tuple[float, float] = (0.1, 0.9)
    center_weight: float = 1.0
    edge_weight: float = 0.2
    enable_color_filtering: bool = False
    min_color_variance: float = 200.0

    # Aliases for backward compatibility
    @property
    def motion_threshold(self) -> int:
        return self.threshold

    @property
    def consecutive_detections_required(self) -> int:
        return self.consecutive_required


class LocationConfig(BaseSettings):
    """Location configuration for sun calculations."""
    model_config = SettingsConfigDict(env_prefix='LOCATION_')

    latitude: float = 50.7374  # Bonn, Germany
    longitude: float = 7.0982
    timezone: str = "Europe/Berlin"

    @field_validator('latitude')
    @classmethod
    def validate_lat(cls, v):
        if not (-90.0 <= v <= 90.0):
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator('longitude')
    @classmethod
    def validate_lon(cls, v):
        if not (-180.0 <= v <= 180.0):
            raise ValueError("Longitude must be between -180 and 180")
        return v


class PerformanceConfig(BaseSettings):
    """Performance and resource management configuration."""
    model_config = SettingsConfigDict(env_prefix='PERFORMANCE_')

    cooldown_period: float = 30.0
    memory_threshold: float = 0.8
    processing_timeout: float = 30.0
    max_images: int = 100
    idle_sleep: float = 0.05
    cooldown_sleep: float = 0.1
    error_sleep: float = 5.0
    cleanup_days: int = 30
    capture_delay: float = 0.75
    daylight_only: bool = True
    enable_multi_frame: bool = True
    multi_frame_count: int = 5
    multi_frame_interval: float = 0.1
    min_sharpness_threshold: float = 15.0
    motion_aware_selection: bool = True


class StorageConfig(BaseSettings):
    """Storage and file management configuration."""
    model_config = SettingsConfigDict(env_prefix='STORAGE_')

    data_dir: Path = Path("data")
    database_path: str = "data/detections.db"
    image_prefix: str = "capture_"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_directories(self):
        """Create required directories."""
        for directory in [self.data_dir, self.image_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class SpeciesConfig(BaseSettings):
    """Species identification configuration."""
    model_config = SettingsConfigDict(env_prefix='SPECIES_')

    model_version: str = "v4.0.1a"
    country_code: str = "DEU"
    admin1_region: str = "NW"
    min_detection_confidence: float = 0.2
    min_classification_confidence: float = 0.5
    unknown_species_threshold: float = 0.5
    processing_timeout: float = 30.0
    return_top_k: int = 5
    crop_padding: float = 0.1

    @field_validator('model_version')
    @classmethod
    def validate_model(cls, v):
        if v not in ["v4.0.1a", "v4.0.1b"]:
            raise ValueError(f"Invalid model version: {v}")
        return v


class Config(BaseSettings):
    """Main configuration aggregating all sections."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    def __init__(self, _env_file='.env', **kwargs):
        """Initialize config, optionally disabling .env file loading."""
        super().__init__(_env_file=_env_file, **kwargs)

    telegram_bot_token: str
    telegram_chat_id: str

    # Sub-configurations (loaded separately since pydantic-settings doesn't nest well)
    _camera: Optional[CameraConfig] = None
    _motion: Optional[MotionConfig] = None
    _performance: Optional[PerformanceConfig] = None
    _storage: Optional[StorageConfig] = None
    _species: Optional[SpeciesConfig] = None
    _location: Optional[LocationConfig] = None

    def model_post_init(self, __context):
        """Initialize sub-configurations and ensure directories exist."""
        self._camera = CameraConfig()
        self._motion = MotionConfig()
        self._performance = PerformanceConfig()
        self._storage = StorageConfig()
        self._species = SpeciesConfig()
        self._location = LocationConfig()
        self._storage.ensure_directories()
        Path(self._storage.database_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Configuration loaded successfully")

    @property
    def camera(self) -> CameraConfig:
        return self._camera

    @property
    def motion(self) -> MotionConfig:
        return self._motion

    @property
    def performance(self) -> PerformanceConfig:
        return self._performance

    @property
    def storage(self) -> StorageConfig:
        return self._storage

    @property
    def species(self) -> SpeciesConfig:
        return self._species

    @property
    def location(self) -> LocationConfig:
        return self._location

    # Legacy attribute names for compatibility
    @property
    def telegram_token(self) -> str:
        return self.telegram_bot_token

    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            "camera": {
                "main_resolution": self.camera.main_resolution,
                "motion_resolution": self.camera.motion_detection_resolution,
            },
            "motion": {
                "threshold": self.motion.threshold,
                "frame_interval": self.motion.frame_interval,
            },
            "species": {
                "model_version": self.species.model_version,
                "country": self.species.country_code,
                "min_detection_confidence": self.species.min_detection_confidence,
            }
        }

    @classmethod
    def create_test_config(cls) -> 'Config':
        """Create configuration for testing."""
        import os
        os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'test_token')
        os.environ.setdefault('TELEGRAM_CHAT_ID', 'test_chat')
        return cls()


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass
