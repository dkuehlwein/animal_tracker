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
    Config, CameraConfig, MotionConfig,
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

    def test_ae_exposure_mode_default_is_short(self):
        """Default biases AE toward short exposure (dusk fix)."""
        config = CameraConfig()
        assert config.ae_exposure_mode == "short"

    @patch.dict(os.environ, {'CAMERA_AE_EXPOSURE_MODE': 'long'}, clear=False)
    def test_ae_exposure_mode_env_override(self):
        """CAMERA_AE_EXPOSURE_MODE overrides the default (e.g. rollback lever)."""
        config = CameraConfig()
        assert config.ae_exposure_mode == "long"

    def test_invalid_ae_exposure_mode(self):
        """Only normal|short|long are accepted."""
        with pytest.raises(ValidationError):
            CameraConfig(ae_exposure_mode="bogus")


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
        # logs_dir aliases log_dir (single source of truth), which is
        # independent of data_dir and defaults to data/logs.
        assert config.logs_dir == config.log_dir == Path("data/logs")


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

    def test_human_detection_confidence_default(self):
        """Default human_detection_confidence must be 0.3."""
        config = SpeciesConfig()
        assert config.human_detection_confidence == 0.3

    def test_human_detection_confidence_env_override(self, monkeypatch):
        """SPECIES_HUMAN_DETECTION_CONFIDENCE overrides the default."""
        monkeypatch.setenv("SPECIES_HUMAN_DETECTION_CONFIDENCE", "0.6")
        config = SpeciesConfig(_env_file=None)
        assert config.human_detection_confidence == 0.6

    def test_human_detection_confidence_rejects_above_one(self, monkeypatch):
        monkeypatch.setenv("SPECIES_HUMAN_DETECTION_CONFIDENCE", "1.5")
        with pytest.raises(ValidationError):
            SpeciesConfig(_env_file=None)

    def test_human_detection_confidence_rejects_below_zero(self, monkeypatch):
        monkeypatch.setenv("SPECIES_HUMAN_DETECTION_CONFIDENCE", "-0.1")
        with pytest.raises(ValidationError):
            SpeciesConfig(_env_file=None)


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


def test_suppress_human_alerts_defaults_true():
    from config import Config
    cfg = Config.create_test_config()
    assert cfg.performance.suppress_human_alerts is True


def test_suppress_human_alerts_env_override(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_SUPPRESS_HUMAN_ALERTS", "false")
    from config import PerformanceConfig
    assert PerformanceConfig().suppress_human_alerts is False


def test_human_retention_hours_defaults_48():
    from config import Config
    cfg = Config.create_test_config()
    assert cfg.performance.human_retention_hours == 48


def test_human_retention_hours_env_override(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_HUMAN_RETENTION_HOURS", "12")
    from config import PerformanceConfig
    assert PerformanceConfig().human_retention_hours == 12


def test_log_dir_defaults_to_data_logs():
    """log_dir must default to data/logs so rotating file logs survive
    reboots without requiring any configuration (journald history does not)."""
    from config import StorageConfig
    assert StorageConfig().log_dir == Path("data/logs")


def test_log_dir_env_override(monkeypatch):
    monkeypatch.setenv("STORAGE_LOG_DIR", "custom/log/path")
    from config import StorageConfig
    assert StorageConfig().log_dir == Path("custom/log/path")


def test_logs_dir_property_follows_log_dir_override(monkeypatch):
    """logs_dir (used by resource_manager.ensure_directories) must share a
    single source of truth with log_dir, so a STORAGE_LOG_DIR override moves
    both — otherwise the file handler writes to one directory while
    ensure_directories creates another."""
    monkeypatch.setenv("STORAGE_LOG_DIR", "custom/log/path")
    from config import StorageConfig
    assert StorageConfig().logs_dir == Path("custom/log/path")


class TestSceneGateConfig:
    """Scene-unchanged gate config knobs (Task 3, feat/scene-gate)."""

    def test_defaults(self):
        from config import PerformanceConfig
        config = PerformanceConfig()
        assert config.scene_gate_enabled is True
        assert config.scene_gate_similarity_threshold == 0.97
        assert config.scene_gate_ref_count == 3
        assert config.scene_gate_ref_max_age_hours == 6.0

    def test_scene_gate_enabled_env_override(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_ENABLED", "false")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_enabled is False

    def test_scene_gate_similarity_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD", "0.9")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_similarity_threshold == 0.9

    def test_scene_gate_ref_count_env_override(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_REF_COUNT", "5")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_ref_count == 5

    def test_scene_gate_ref_max_age_hours_env_override(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_REF_MAX_AGE_HOURS", "12.5")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_ref_max_age_hours == 12.5

    def test_similarity_threshold_rejects_below_bound(self, monkeypatch):
        # BOUNDS["PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD"] = (0.80, 1.0)
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD", "0.5")
        from config import PerformanceConfig
        with pytest.raises(ValidationError):
            PerformanceConfig(_env_file=None)

    def test_similarity_threshold_rejects_above_bound(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD", "1.5")
        from config import PerformanceConfig
        with pytest.raises(ValidationError):
            PerformanceConfig(_env_file=None)

    def test_similarity_threshold_accepts_lower_bound(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD", "0.80")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_similarity_threshold == 0.80

    def test_similarity_threshold_accepts_upper_bound(self, monkeypatch):
        monkeypatch.setenv("PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD", "1.0")
        from config import PerformanceConfig
        assert PerformanceConfig(_env_file=None).scene_gate_similarity_threshold == 1.0


if __name__ == '__main__':
    pytest.main([__file__])
