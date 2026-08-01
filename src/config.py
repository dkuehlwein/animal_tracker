"""
Configuration management for Wildlife Detector system.
Uses pydantic-settings for type-safe configuration with environment variable overrides.
"""

from pathlib import Path
from typing import Tuple, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import sys as _sys
from pathlib import Path as _Path
# guardrails lives in src/loop; ensure src is importable when config is imported
# standalone (production runs with src on the path; this is belt-and-braces).
_SRC = _Path(__file__).resolve().parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from loop.guardrails import BOUNDS as _BOUNDS  # noqa: E402

logger = logging.getLogger(__name__)

# Sentinel marking "caller did not specify _env_file" so Config.__init__ can
# defer to model_config['env_file'] (which tests patch to None). Module-level so
# it is unaffected by pydantic's class-attribute handling.
_ENV_FILE_UNSET = object()


class CameraConfig(BaseSettings):
    """Camera-specific configuration settings."""
    model_config = SettingsConfigDict(env_prefix='CAMERA_', env_file='.env', extra='ignore')

    main_resolution: Tuple[int, int] = (2028, 1520)  # IMX477 native 4:3 binned mode
    motion_detection_resolution: Tuple[int, int] = (640, 480)
    motion_detection_format: str = "YUV420"
    frame_format: str = "RGB888"  # Main stream format for capture
    frame_duration: int = 100000  # microseconds
    startup_delay: float = 2.0
    exposure_time: Optional[int] = None  # None = auto-exposure
    analogue_gain: Optional[float] = None  # None = auto-gain
    # Dusk-exposure fix: bias AE toward shorter exposure (+ higher gain) so
    # dusk bursts don't fall below the sharpness floor. Only applies when AE
    # is active (exposure_time/analogue_gain are None); mapped to libcamera's
    # AeExposureMode by PiCameraManager. Rollback lever: set to "normal".
    ae_exposure_mode: str = "short"

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
            raise ValueError("Exposure time must be between 100-1000000 μs")
        return v

    @field_validator('analogue_gain')
    @classmethod
    def validate_gain(cls, v):
        if v is not None and (v < 1.0 or v > 8.0):
            raise ValueError("Analogue gain must be between 1.0-8.0")
        return v

    @field_validator('ae_exposure_mode')
    @classmethod
    def validate_ae_exposure_mode(cls, v):
        if v not in ("normal", "short", "long"):
            raise ValueError(f"Invalid AE exposure mode: {v} (must be normal|short|long)")
        return v


class MotionConfig(BaseSettings):
    """Motion detection configuration settings."""
    # Overlay precedence (pydantic-settings): real OS env > deployed_config.env
    # > .env > field defaults. The loop's deploy step renders deployed_config.env;
    # a human OS env var still wins. A missing overlay file is ignored.
    model_config = SettingsConfigDict(
        env_prefix='MOTION_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )

    threshold: int = 2000
    min_contour_area: int = 50
    background_history: int = 500
    background_threshold: int = 40
    frame_interval: float = 0.2  # 5 FPS
    consecutive_required: int = 2
    blur_kernel_size: int = 5
    central_region_bounds: Tuple[float, float] = (0.1, 0.9)
    center_weight: float = 1.0
    edge_weight: float = 0.2
    enable_color_filtering: bool = False
    min_color_variance: float = 200.0
    warmup_seconds: float = 300.0  # Suppress detection while MOG2 learns the scene

    # Aliases for backward compatibility
    @property
    def motion_threshold(self) -> int:
        return self.threshold

    @property
    def consecutive_detections_required(self) -> int:
        return self.consecutive_required

    @field_validator('threshold')
    @classmethod
    def validate_threshold_bounds(cls, v):
        low, high = _BOUNDS["MOTION_THRESHOLD"]
        if not (low <= v <= high):
            raise ValueError(
                f"MOTION_THRESHOLD={v} out of allowed bounds [{low}, {high}]"
            )
        return v


class LocationConfig(BaseSettings):
    """Location configuration for sun calculations."""
    model_config = SettingsConfigDict(env_prefix='LOCATION_', env_file='.env', extra='ignore')

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
    model_config = SettingsConfigDict(
        env_prefix='PERFORMANCE_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )

    cooldown_period: float = 30.0
    memory_threshold: float = 0.8
    processing_timeout: float = 30.0
    max_images: int = 300
    idle_sleep: float = 0.05
    cooldown_sleep: float = 0.1
    error_sleep: float = 5.0
    cleanup_days: int = 30
    capture_delay: float = 0.75
    daylight_only: bool = True
    enable_multi_frame: bool = True
    multi_frame_count: int = 5
    multi_frame_interval: float = 0.1
    min_sharpness_threshold: float = 11.0
    motion_aware_selection: bool = True
    send_annotated_image: bool = False  # Debug: send motion detection overlay alongside original
    review_prefix_enabled: bool = True  # Prefix likely-FP (NO_ANIMAL/UNCLASSIFIABLE) captions with 🔍 REVIEW header
    suppress_human_alerts: bool = True  # Skip Telegram notification for HUMAN-status detections (still DB-logged)
    human_retention_hours: int = 48  # Purge saved photos of HUMAN-status detections after this many hours (DB row kept)

    # Timelapse FN-audit channel (ADR-004 Phase 1): low-rate independent capture
    enable_timelapse: bool = True
    timelapse_interval: float = 20.0  # seconds between saved frames
    timelapse_max_files: int = 10000  # ~2 days @ 20s; oldest pruned beyond this

    # Scene-unchanged gate: mute captures whose burst frame is near-identical
    # to a recent reference frame (empty-scene FP reduction; see scene_gate.py).
    # Task 5 (offline validation, 2026-07-17): disabled by default. The
    # labeled corpus has ZERO human 'animal'/'animal_wrong_id' review-class
    # rows whose frame still exists on disk (all 17 predate the ~100-burst
    # retention window vs. 53 on-disk review-class frames) — no threshold
    # can be picked with any evidence it won't mute a real animal. Ship
    # disabled until more labeled animal frames survive retention long
    # enough to be replayed; see scripts/validate_scene_gate.py and
    # .superpowers/sdd/task-5-report.md. Flip to True (with a threshold set
    # from a re-run of the validation script) once that data exists.
    scene_gate_enabled: bool = False
    # Conservative placeholder — kept at 0.97 pending a validated default
    # (see scene_gate_enabled note above).
    scene_gate_similarity_threshold: float = 0.97
    scene_gate_ref_count: int = 3
    scene_gate_ref_max_age_hours: float = 6.0

    # Blur-mute (below-floor + no-animal) only fires when best-frame mean
    # luma >= this; below it, darkness (not blur) explains the low score,
    # so the burst flows to REVIEW instead of being muted (exp #8,
    # FN-safe — see 2026-07-14 muted dusk blackbird, frame luma 67.8).
    blur_mute_min_luma: float = 70.0

    # REVIEW-channel sampling gate: only this fraction of review-class
    # (NO_ANIMAL/UNCLASSIFIABLE) bursts that survive the Human/Blur/Scene
    # mute gates are actually sent to Telegram — everything is still
    # species-ID'd and DB-logged regardless (see
    # wildlife_system.is_review_sampled_out). 1.0 = send everything
    # (rollback lever); 0.0 = send nothing.
    review_sample_rate: float = 0.25

    # Human-proximity mute gate: mute review-class (NO_ANIMAL/UNCLASSIFIABLE)
    # bursts that land within this many seconds after the most recent
    # HUMAN-status detection. MegaDetector scores extreme close-up /
    # motion-blurred partial human bodies at ~0.02-0.15 person confidence, so
    # such bursts slip past the Human/Privacy Gate as no_animal and leak a
    # recognizable person to REVIEW (2026-07-27, detection ids 3544/3553/3554,
    # each 76-108s after a correctly-classified human burst). Validated
    # against all 12 human-labelled animal/animal_wrong_id review-class rows
    # since the human gate went live (2026-07-08): the closest is 329s from a
    # preceding human-status burst, so a 120s window costs zero known false
    # negatives. 0.0 disables the gate (rollback lever).
    human_proximity_window_seconds: float = 120.0

    # Human-density condition (2026-07-28, exp #11 mechanism extension):
    # OR-ed onto the human-proximity gate above. Tonight's adjudication found
    # recognizable-person review-class bursts OUTSIDE the proximity window
    # (gaps of 432s and 732s past the last human burst) during a long
    # gardening session — a single "time since last human" window can't catch
    # a garden that stays occupied longer than the window. This condition
    # mutes instead when at least `human_density_count` HUMAN-status
    # detections occurred in the trailing `human_density_window_seconds`
    # ("the garden is occupied", not just "a human was just here").
    # `human_density_count=0` disables this condition (rollback lever); the
    # existing window condition is untouched and still fires on its own.
    human_density_window_seconds: float = 1800.0
    human_density_count: int = 8

    # Leading-edge fix (2026-07-31): the human-proximity gate above is
    # backward-looking only (it mutes AFTER a HUMAN-status detection), so it
    # can never catch the LEADING EDGE of a human visit — burst 3909
    # (2026-07-31, 18:22:42) was sent to REVIEW as no_animal 81s BEFORE the
    # visit's first HUMAN burst (18:24:03), with a clearly recognisable face
    # in its saved frames (two prior instances: 75s, 51s gaps). Rather than
    # holding every review-class burst hostage indefinitely, delay its
    # Telegram send by this many seconds; if a HUMAN-status detection lands
    # within that window, cancel the send instead (see
    # wildlife_system._deferred_review_send). 0.0 disables deferral
    # (rollback lever) — reviews send immediately, as before this fix.
    review_defer_seconds: float = 240.0

    # Same fix, storage side (Part B): today only detection_status='human'
    # bursts get their photos purged after human_retention_hours. A
    # review-class burst that really contains a person (misclassified as
    # no_animal, e.g. the leading-edge leak above) keeps recognisable frames
    # on disk for the full ~300-burst rotation otherwise. Extends the
    # purge SYMMETRICALLY in time — a no_animal/unclassifiable burst within
    # this many seconds of a HUMAN-status detection (before OR after) is
    # purged the same way (see resource_manager.StorageManager.purge_human_bursts
    # and database_manager.get_human_adjacent_review_detections). 0.0
    # disables (rollback lever).
    human_retention_proximity_seconds: float = 240.0

    @field_validator('review_defer_seconds')
    @classmethod
    def validate_review_defer_seconds_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_REVIEW_DEFER_SECONDS"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_REVIEW_DEFER_SECONDS={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('human_retention_proximity_seconds')
    @classmethod
    def validate_human_retention_proximity_seconds_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('scene_gate_similarity_threshold')
    @classmethod
    def validate_scene_gate_similarity_threshold_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('blur_mute_min_luma')
    @classmethod
    def validate_blur_mute_min_luma_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_BLUR_MUTE_MIN_LUMA"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_BLUR_MUTE_MIN_LUMA={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('review_sample_rate')
    @classmethod
    def validate_review_sample_rate_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_REVIEW_SAMPLE_RATE"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_REVIEW_SAMPLE_RATE={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('human_proximity_window_seconds')
    @classmethod
    def validate_human_proximity_window_seconds_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('human_density_window_seconds')
    @classmethod
    def validate_human_density_window_seconds_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('human_density_count')
    @classmethod
    def validate_human_density_count_bounds(cls, v):
        low, high = _BOUNDS["PERFORMANCE_HUMAN_DENSITY_COUNT"]
        if not (low <= v <= high):
            raise ValueError(
                f"PERFORMANCE_HUMAN_DENSITY_COUNT={v} out of allowed bounds [{low}, {high}]"
            )
        return v


class StorageConfig(BaseSettings):
    """Storage and file management configuration."""
    model_config = SettingsConfigDict(env_prefix='STORAGE_', env_file='.env', extra='ignore')

    data_dir: Path = Path("data")
    database_path: str = "data/detections.db"
    image_prefix: str = "capture_"
    # Rotating file log destination (Task 2: journald history does not
    # survive a Pi reboot, so INFO+ logs are also written to disk here).
    log_dir: Path = Path("data/logs")

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def logs_dir(self) -> Path:
        # Alias for log_dir (single source of truth) so existing callers
        # (resource_manager.ensure_directories) follow STORAGE_LOG_DIR too.
        return self.log_dir

    def ensure_directories(self):
        """Create required directories."""
        for directory in [self.data_dir, self.image_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class SpeciesConfig(BaseSettings):
    """Species identification configuration."""
    model_config = SettingsConfigDict(
        env_prefix='SPECIES_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )

    model_version: str = "v4.0.1a"
    country_code: str = "DEU"
    admin1_region: str = "NW"
    min_detection_confidence: float = 0.2
    min_classification_confidence: float = 0.5
    unknown_species_threshold: float = 0.5
    processing_timeout: float = 30.0
    return_top_k: int = 5
    crop_padding: float = 0.1
    # Human/privacy gate: MegaDetector person-box confidence at/above this
    # value fires DetectionStatus.HUMAN (see species_identifier._parse_predictions).
    human_detection_confidence: float = 0.3

    @field_validator('model_version')
    @classmethod
    def validate_model(cls, v):
        if v not in ["v4.0.1a", "v4.0.1b"]:
            raise ValueError(f"Invalid model version: {v}")
        return v

    @field_validator('unknown_species_threshold')
    @classmethod
    def validate_unknown_threshold_bounds(cls, v):
        low, high = _BOUNDS["SPECIES_UNKNOWN_SPECIES_THRESHOLD"]
        if not (low <= v <= high):
            raise ValueError(
                f"SPECIES_UNKNOWN_SPECIES_THRESHOLD={v} out of allowed bounds [{low}, {high}]"
            )
        return v

    @field_validator('human_detection_confidence')
    @classmethod
    def validate_human_detection_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"SPECIES_HUMAN_DETECTION_CONFIDENCE={v} out of allowed bounds [0.0, 1.0]"
            )
        return v


class Config(BaseSettings):
    """Main configuration aggregating all sections."""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    def __init__(self, _env_file=_ENV_FILE_UNSET, **kwargs):
        """Initialize config, optionally disabling .env file loading.

        When ``_env_file`` is not given explicitly we defer to the class's
        ``model_config['env_file']`` (normally ``.env``). Tests patch that to
        ``None`` to keep the production ``.env`` from leaking into assertions, so
        the default must honour the patched value rather than hard-coding
        ``'.env'``.
        """
        if _env_file is _ENV_FILE_UNSET:
            _env_file = type(self).model_config.get('env_file', '.env')
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
        """Create configuration for testing.

        Isolation note: tests must never read the production ``.env`` (it leaks
        operational overrides such as ``MOTION_THRESHOLD`` into assertions) nor
        resolve storage to the real ``data/`` directory (a test that writes or
        unlinks would corrupt production data). Both invariants are enforced by
        ``tests/conftest.py``, which disables ``.env`` loading and redirects
        ``STORAGE_DATA_DIR`` / ``STORAGE_DATABASE_PATH`` to a temp directory for
        the whole session before any config is built. This factory only supplies
        the required telegram credentials and a zero warmup window.
        """
        import os
        os.environ.setdefault('TELEGRAM_BOT_TOKEN', 'test_token')
        os.environ.setdefault('TELEGRAM_CHAT_ID', 'test_chat')
        os.environ.setdefault('MOTION_WARMUP_SECONDS', '0')
        return cls()


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass
