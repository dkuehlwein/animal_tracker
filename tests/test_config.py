"""
Unit tests for configuration system.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

import sys
sys.path.append('src')

from config import (
    Config, ConfigurationError, CameraConfig, MotionConfig,
    PerformanceConfig, StorageConfig, SpeciesConfig
)


class TestCameraConfig:
    """Test camera configuration validation."""

    def test_valid_camera_config(self):
        """Test valid camera configuration."""
        config = CameraConfig()
        assert config.main_resolution == (1920, 1080)
        assert config.motion_detection_format == "YUV420"

    def test_invalid_format(self):
        """Test invalid format validation."""
        with pytest.raises(ValidationError):
            CameraConfig(motion_detection_format="INVALID")

    def test_invalid_exposure_time(self):
        """Test invalid exposure time validation."""
        with pytest.raises(ValidationError):
            CameraConfig(exposure_time=50)  # Below 100

    def test_invalid_analogue_gain(self):
        """Test invalid analogue gain validation."""
        with pytest.raises(ValidationError):
            CameraConfig(analogue_gain=10.0)  # Above 8.0


class TestMotionConfig:
    """Test motion detection configuration validation."""

    def test_valid_motion_config(self):
        """Test valid motion configuration."""
        config = MotionConfig()
        assert config.threshold == 2000
        assert config.motion_threshold == 2000  # Alias
        assert config.central_region_bounds == (0.2, 0.8)

    def test_backward_compatibility_aliases(self):
        """Test backward compatibility aliases work."""
        config = MotionConfig()
        assert config.motion_threshold == config.threshold
        assert config.consecutive_detections_required == config.consecutive_required


class TestPerformanceConfig:
    """Test performance configuration validation."""

    def test_valid_performance_config(self):
        """Test valid performance configuration."""
        config = PerformanceConfig()
        assert config.memory_threshold == 0.8
        assert config.max_images == 100


class TestStorageConfig:
    """Test storage configuration validation."""

    def test_valid_storage_config(self):
        """Test valid storage configuration."""
        config = StorageConfig()
        assert config.data_dir == Path("data")
        assert config.image_prefix == "capture_"

    def test_derived_paths(self):
        """Test derived path properties."""
        config = StorageConfig(data_dir=Path("custom"))
        assert config.image_dir == Path("custom/images")
        assert config.logs_dir == Path("custom/logs")


class TestSpeciesConfig:
    """Test species identification configuration."""

    def test_valid_species_config(self):
        """Test valid species configuration."""
        config = SpeciesConfig()
        assert config.model_version == "v4.0.1a"
        assert config.min_detection_confidence == 0.2

    def test_invalid_model_version(self):
        """Test invalid model version validation."""
        with pytest.raises(ValidationError):
            SpeciesConfig(model_version="v1.0.0")


class TestConfig:
    """Test main configuration class."""

    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token_123',
        'TELEGRAM_CHAT_ID': 'test_chat_123'
    }, clear=True)
    def test_valid_config(self):
        """Test valid configuration loading."""
        config = Config()

        assert config.telegram_token == 'test_token_123'
        assert config.telegram_chat_id == 'test_chat_123'
        assert isinstance(config.camera, CameraConfig)
        assert isinstance(config.motion, MotionConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.storage, StorageConfig)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_telegram_token(self):
        """Test missing Telegram token."""
        with pytest.raises(ValidationError):
            Config(_env_file=None)

    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test'}, clear=True)
    def test_missing_telegram_chat_id(self):
        """Test missing Telegram chat ID."""
        with pytest.raises(ValidationError):
            Config(_env_file=None)

    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat',
        'CAMERA_MAIN_RESOLUTION': '[3840, 2160]'  # Pydantic uses JSON for tuples
    }, clear=True)
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config = Config(_env_file=None)
        assert config.camera.main_resolution == (3840, 2160)

    def test_create_test_config(self):
        """Test test configuration creation."""
        config = Config.create_test_config()
        assert config.telegram_token == 'test_token'
        assert config.telegram_chat_id == 'test_chat'

    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat'
    }, clear=True)
    def test_get_summary(self):
        """Test configuration summary."""
        config = Config()
        summary = config.get_summary()

        assert 'camera' in summary
        assert 'motion' in summary
        assert 'species' in summary

        assert summary['camera']['main_resolution'] == (1920, 1080)
        assert summary['motion']['threshold'] == 2000


if __name__ == '__main__':
    pytest.main([__file__])
