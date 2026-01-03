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
    # Exposure control (motion freeze settings)
    # None = auto-exposure (adapts to lighting, but may have motion blur)
    # Manual: Bright sun: 2000μs + 2.5x | Cloudy: 15000μs + 8.0x | Current: auto
    exposure_time: Optional[int] = None  # microseconds, None for auto-exposure
    analogue_gain: Optional[float] = None  # Sensor gain (1.0-8.0), None for auto
    
    def __post_init__(self):
        """Validate camera configuration."""
        if not self._validate_resolution(self.main_resolution):
            raise ValueError(f"Invalid main resolution: {self.main_resolution}")
        if not self._validate_resolution(self.motion_detection_resolution):
            raise ValueError(f"Invalid motion detection resolution: {self.motion_detection_resolution}")
        if self.motion_detection_format not in ["YUV420", "RGB888"]:
            raise ValueError(f"Invalid motion detection format: {self.motion_detection_format}")
        if self.exposure_time is not None and (self.exposure_time < 100 or self.exposure_time > 1000000):
            raise ValueError(f"Exposure time must be between 100-1000000 μs (got {self.exposure_time})")
        if self.analogue_gain is not None and (self.analogue_gain < 1.0 or self.analogue_gain > 8.0):
            raise ValueError(f"Analogue gain must be between 1.0-8.0 (got {self.analogue_gain})")
    
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
    background_threshold: int = 40  # Increased from 20 to reduce noise sensitivity
    frame_interval: float = 0.2  # 5 FPS
    consecutive_detections_required: int = 2
    blur_kernel_size: int = 5  # Gaussian blur to reduce camera noise
    central_region_bounds: Tuple[float, float] = (0.2, 0.8)
    center_weight: float = 1.0
    edge_weight: float = 0.2

    # Color-based motion filtering (reduces false alarms from uniform-color leaves)
    # When enabled, captures RGB frames instead of grayscale and filters out motion
    # from uniform-color objects (like waving leaves). Slower but more accurate.
    # Disabled by default as motion-aware frame selection handles most false alarms.
    enable_color_filtering: bool = False
    min_color_variance: float = 200.0  # Minimum color variance to consider motion valid

    def __post_init__(self):
        """Validate motion detection configuration."""
        if self.motion_threshold <= 0:
            raise ValueError("Motion threshold must be positive")
        if not (0.0 < self.central_region_bounds[0] < self.central_region_bounds[1] < 1.0):
            raise ValueError("Invalid central region bounds")
        if self.center_weight <= 0 or self.edge_weight < 0:
            raise ValueError("Invalid weight values")
        if self.min_color_variance < 0:
            raise ValueError("Minimum color variance must be non-negative")


@dataclass(frozen=True)
class LocationConfig:
    """Location configuration for sun calculations."""
    latitude: float = 50.7374  # Bonn, Germany
    longitude: float = 7.0982
    timezone: str = "Europe/Berlin"
    
    def __post_init__(self):
        """Validate location configuration."""
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError("Longitude must be between -180 and 180")


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
    capture_delay: float = 0.75  # Delay before capturing high-res photo (allows animal to settle)
    daylight_only: bool = True  # Only track during daylight hours

    # Multi-frame burst capture settings
    enable_multi_frame: bool = True  # Enable multi-frame burst capture with sharpness analysis
    multi_frame_count: int = 5  # Number of frames to capture in burst
    multi_frame_interval: float = 0.1  # Interval between burst frames in seconds
    min_sharpness_threshold: float = 15.0  # Minimum acceptable sharpness score (scene-dependent, 15-30 typical for outdoor)
    motion_aware_selection: bool = True  # Use reference-based scoring to select frames with most subject content

    def __post_init__(self):
        """Validate performance configuration."""
        if not (0.0 < self.memory_threshold < 1.0):
            raise ValueError("Memory threshold must be between 0 and 1")
        if self.max_images <= 0:
            raise ValueError("Max images must be positive")
        if self.multi_frame_count < 1:
            raise ValueError("Multi-frame count must be at least 1")
        if self.multi_frame_interval < 0:
            raise ValueError("Multi-frame interval must be non-negative")
        if self.min_sharpness_threshold < 0:
            raise ValueError("Minimum sharpness threshold must be non-negative")


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


@dataclass(frozen=True)
class SpeciesConfig:
    """Species identification configuration."""
    # Model settings
    model_version: str = "v4.0.1a"  # always-crop variant
    use_ensemble: bool = True  # Use detector + classifier

    # Geographic filtering
    country_code: str = "DEU"  # ISO 3166-1 Alpha-3 for Germany
    admin1_region: str = "NW"  # North Rhine-Westphalia (Bonn)

    # Confidence thresholds
    min_detection_confidence: float = 0.6  # MegaDetector threshold
    min_classification_confidence: float = 0.5  # Species classification threshold
    unknown_species_threshold: float = 0.5  # Below this = "Unknown species"

    # Performance settings
    model_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "speciesnet")
    enable_gpu: bool = False  # Pi 5 doesn't have NVIDIA GPU
    processing_timeout: float = 30.0  # Max time for identification

    # Behavior settings
    return_top_k: int = 5  # Return top 5 species predictions
    crop_padding: float = 0.1  # Padding around detected objects

    def __post_init__(self):
        """Validate species configuration."""
        if not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ValueError("Detection confidence must be between 0 and 1")
        if not (0.0 <= self.min_classification_confidence <= 1.0):
            raise ValueError("Classification confidence must be between 0 and 1")
        if not (0.0 <= self.unknown_species_threshold <= 1.0):
            raise ValueError("Unknown species threshold must be between 0 and 1")
        if self.model_version not in ["v4.0.1a", "v4.0.1b"]:
            raise ValueError(f"Invalid model version: {self.model_version}")


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
        self.species = self._load_species_config()
        self.location = self._load_location_config()

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
        # Use dataclass defaults, override only if env vars are set
        defaults = CameraConfig.__dataclass_fields__

        # Parse exposure settings (empty string or missing = use dataclass default)
        exposure_time_str = os.getenv("CAMERA_EXPOSURE_TIME")
        if exposure_time_str == "":
            exposure_time = None  # Empty string means auto-exposure
        elif exposure_time_str:
            exposure_time = int(exposure_time_str)  # Use env value
        else:
            exposure_time = defaults['exposure_time'].default  # Use dataclass default

        analogue_gain_str = os.getenv("CAMERA_ANALOGUE_GAIN")
        if analogue_gain_str == "":
            analogue_gain = None  # Empty string means auto-gain
        elif analogue_gain_str:
            analogue_gain = float(analogue_gain_str)  # Use env value
        else:
            analogue_gain = defaults['analogue_gain'].default  # Use dataclass default

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
            ),
            exposure_time=exposure_time,
            analogue_gain=analogue_gain
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
            ),
            enable_color_filtering=self._get_optional_env(
                "MOTION_ENABLE_COLOR_FILTERING", "False", lambda x: x.lower() in ("true", "1", "yes")
            ),
            min_color_variance=self._get_optional_env(
                "MOTION_MIN_COLOR_VARIANCE", "200.0", float
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
            ),
            capture_delay=self._get_optional_env(
                "PERFORMANCE_CAPTURE_DELAY", "0.75", float
            ),
            daylight_only=self._get_optional_env(
                "DAYLIGHT_ONLY", "True", lambda x: x.lower() in ("true", "1", "yes")
            )
        )
    
    def _load_location_config(self) -> LocationConfig:
        """Load location configuration with environment overrides."""
        return LocationConfig(
            latitude=self._get_optional_env(
                "LOCATION_LATITUDE", "50.7374", float
            ),
            longitude=self._get_optional_env(
                "LOCATION_LONGITUDE", "7.0982", float
            ),
            timezone=self._get_optional_env(
                "LOCATION_TIMEZONE", "Europe/Berlin"
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

    def _load_species_config(self) -> SpeciesConfig:
        """Load species identification configuration with environment overrides."""
        return SpeciesConfig(
            model_version=self._get_optional_env(
                "SPECIES_MODEL_VERSION", "v4.0.1a"
            ),
            country_code=self._get_optional_env(
                "SPECIES_COUNTRY_CODE", "DEU"
            ),
            admin1_region=self._get_optional_env(
                "SPECIES_REGION", "NW"
            ),
            min_detection_confidence=self._get_optional_env(
                "SPECIES_MIN_DETECTION_CONF", "0.6", float
            ),
            min_classification_confidence=self._get_optional_env(
                "SPECIES_MIN_CLASS_CONF", "0.5", float
            ),
            unknown_species_threshold=self._get_optional_env(
                "SPECIES_UNKNOWN_THRESHOLD", "0.5", float
            ),
            processing_timeout=self._get_optional_env(
                "SPECIES_PROCESSING_TIMEOUT", "30.0", float
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
            },
            "species": {
                "model_version": self.species.model_version,
                "country": self.species.country_code,
                "region": self.species.admin1_region,
                "min_detection_confidence": self.species.min_detection_confidence,
                "unknown_threshold": self.species.unknown_species_threshold
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