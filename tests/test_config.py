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
        assert config.main_resolution == (2028, 1520)
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
        assert config.central_region_bounds == (0.1, 0.9)

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

    @patch.dict(os.environ, {}, clear=True)
    def test_valid_storage_config(self):
        """Test valid storage configuration.

        Cleared environment so the genuine class default is observed (the test
        harness in conftest.py redirects STORAGE_DATA_DIR to a temp directory for
        isolation; here we assert the documented production default instead).
        """
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

        assert summary['camera']['main_resolution'] == (2028, 1520)
        assert summary['motion']['threshold'] == 2000


class TestOverlayAndBounds:
    """ADR-004 Phase 4: deployed_config.env overlay + bounds validators."""

    def _write_overlay(self, tmp_path, body: str) -> str:
        overlay = tmp_path / "deployed_config.env"
        overlay.write_text(body)
        return str(overlay)

    def _empty_env(self, tmp_path) -> str:
        """Return path to an empty .env file (stands in for None in the tuple)."""
        p = tmp_path / "empty.env"
        p.write_text("")
        return str(p)

    def test_overlay_overrides_defaults(self, tmp_path, monkeypatch):
        # No real OS env for the key; overlay file sets it → overlay wins over default.
        # Adaptation: pydantic-settings rejects None in a tuple; use an empty temp
        # .env as the first element instead. Behaviour under test is identical.
        monkeypatch.delenv("MOTION_THRESHOLD", raising=False)
        overlay = self._write_overlay(tmp_path, "MOTION_THRESHOLD=2500\n")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(self._empty_env(tmp_path), overlay))
        assert cfg.threshold == 2500

    def test_os_env_overrides_overlay(self, tmp_path, monkeypatch):
        # Real OS env beats the overlay file (manual override preserved).
        monkeypatch.setenv("MOTION_THRESHOLD", "3000")
        overlay = self._write_overlay(tmp_path, "MOTION_THRESHOLD=2500\n")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(self._empty_env(tmp_path), overlay))
        assert cfg.threshold == 3000

    def test_missing_overlay_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOTION_THRESHOLD", raising=False)
        missing = str(tmp_path / "does_not_exist.env")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(self._empty_env(tmp_path), missing))
        assert cfg.threshold == 2000  # documented default

    def test_out_of_range_motion_threshold_raises(self, monkeypatch):
        from pydantic import ValidationError
        from config import MotionConfig
        # 100000 is outside guardrails.BOUNDS["MOTION_THRESHOLD"] = (200, 8000).
        monkeypatch.setenv("MOTION_THRESHOLD", "100000")
        with pytest.raises(ValidationError):
            MotionConfig(_env_file=None)

    def test_in_range_motion_threshold_ok(self, monkeypatch):
        from config import MotionConfig
        monkeypatch.setenv("MOTION_THRESHOLD", "2500")
        cfg = MotionConfig(_env_file=None)
        assert cfg.threshold == 2500

    def test_out_of_range_unknown_threshold_raises(self, monkeypatch):
        from pydantic import ValidationError
        from config import SpeciesConfig
        # 1.5 is outside (0.3, 0.95).
        # field 'unknown_species_threshold' + prefix 'SPECIES_' → env var name:
        monkeypatch.setenv("SPECIES_UNKNOWN_SPECIES_THRESHOLD", "1.5")
        with pytest.raises(ValidationError):
            SpeciesConfig(_env_file=None)


def test_review_prefix_enabled_defaults_true():
    from config import Config
    cfg = Config.create_test_config()
    assert cfg.performance.review_prefix_enabled is True


def test_review_prefix_enabled_env_override(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_REVIEW_PREFIX_ENABLED", "false")
    from config import PerformanceConfig
    assert PerformanceConfig().review_prefix_enabled is False


if __name__ == '__main__':
    pytest.main([__file__])
