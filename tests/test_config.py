"""
Unit tests for configuration system.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append('src')

from config import (
    Config, ConfigurationError, CameraConfig, MotionConfig,
    PerformanceConfig, StorageConfig
)


class TestCameraConfig:
    """Test camera configuration validation."""
    
    def test_valid_camera_config(self):
        """Test valid camera configuration."""
        config = CameraConfig()
        assert config.main_resolution == (1920, 1080)
        assert config.motion_detection_format == "YUV420"
    
    def test_invalid_resolution(self):
        """Test invalid resolution validation."""
        with pytest.raises(ValueError, match="Invalid main resolution"):
            CameraConfig(main_resolution=(0, 1080))
        
        with pytest.raises(ValueError, match="Invalid motion detection resolution"):
            CameraConfig(motion_detection_resolution=(-640, 480))
    
    def test_invalid_format(self):
        """Test invalid format validation."""
        with pytest.raises(ValueError, match="Invalid motion detection format"):
            CameraConfig(motion_detection_format="INVALID")


class TestMotionConfig:
    """Test motion detection configuration validation."""
    
    def test_valid_motion_config(self):
        """Test valid motion configuration."""
        config = MotionConfig()
        assert config.motion_threshold == 2000
        assert config.central_region_bounds == (0.2, 0.8)
    
    def test_invalid_threshold(self):
        """Test invalid motion threshold."""
        with pytest.raises(ValueError, match="Motion threshold must be positive"):
            MotionConfig(motion_threshold=0)
        
        with pytest.raises(ValueError, match="Motion threshold must be positive"):
            MotionConfig(motion_threshold=-100)
    
    def test_invalid_bounds(self):
        """Test invalid central region bounds."""
        with pytest.raises(ValueError, match="Invalid central region bounds"):
            MotionConfig(central_region_bounds=(0.8, 0.2))  # Reversed
        
        with pytest.raises(ValueError, match="Invalid central region bounds"):
            MotionConfig(central_region_bounds=(0.0, 0.5))  # Start at 0
        
        with pytest.raises(ValueError, match="Invalid central region bounds"):
            MotionConfig(central_region_bounds=(0.5, 1.0))  # End at 1
    
    def test_invalid_weights(self):
        """Test invalid weight values."""
        with pytest.raises(ValueError, match="Invalid weight values"):
            MotionConfig(center_weight=0)
        
        with pytest.raises(ValueError, match="Invalid weight values"):
            MotionConfig(edge_weight=-0.1)


class TestPerformanceConfig:
    """Test performance configuration validation."""
    
    def test_valid_performance_config(self):
        """Test valid performance configuration."""
        config = PerformanceConfig()
        assert config.memory_threshold == 0.8
        assert config.max_images == 100
    
    def test_invalid_memory_threshold(self):
        """Test invalid memory threshold."""
        with pytest.raises(ValueError, match="Memory threshold must be between 0 and 1"):
            PerformanceConfig(memory_threshold=0.0)
        
        with pytest.raises(ValueError, match="Memory threshold must be between 0 and 1"):
            PerformanceConfig(memory_threshold=1.0)
        
        with pytest.raises(ValueError, match="Memory threshold must be between 0 and 1"):
            PerformanceConfig(memory_threshold=1.5)
    
    def test_invalid_max_images(self):
        """Test invalid max images."""
        with pytest.raises(ValueError, match="Max images must be positive"):
            PerformanceConfig(max_images=0)
        
        with pytest.raises(ValueError, match="Max images must be positive"):
            PerformanceConfig(max_images=-10)


class TestStorageConfig:
    """Test storage configuration validation."""
    
    @patch('pathlib.Path.mkdir')
    def test_valid_storage_config(self, mock_mkdir):
        """Test valid storage configuration."""
        config = StorageConfig()
        assert config.data_dir == Path("data")
        assert config.image_prefix == "capture_"
        # Should call mkdir for each directory
        assert mock_mkdir.call_count == 3


class TestConfig:
    """Test main configuration class."""
    
    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token_123',
        'TELEGRAM_CHAT_ID': 'test_chat_123'
    })
    @patch('pathlib.Path.mkdir')
    def test_valid_config(self, mock_mkdir):
        """Test valid configuration loading."""
        config = Config()
        
        assert config.telegram_token == 'test_token_123'
        assert config.telegram_chat_id == 'test_chat_123'
        assert isinstance(config.camera, CameraConfig)
        assert isinstance(config.motion, MotionConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.storage, StorageConfig)
    
    def test_missing_telegram_token(self):
        """Test missing Telegram token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="TELEGRAM_BOT_TOKEN"):
                Config()
    
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test'})
    def test_missing_telegram_chat_id(self):
        """Test missing Telegram chat ID."""
        with pytest.raises(ConfigurationError, match="TELEGRAM_CHAT_ID"):
            Config()
    
    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat',
        'CAMERA_MAIN_RESOLUTION': '3840x2160'
    })
    @patch('pathlib.Path.mkdir')
    def test_environment_overrides(self, mock_mkdir):
        """Test environment variable overrides."""
        config = Config()
        assert config.camera.main_resolution == (3840, 2160)
    
    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat',
        'CAMERA_MAIN_RESOLUTION': 'invalid_format'
    })
    @patch('pathlib.Path.mkdir')
    def test_invalid_resolution_format(self, mock_mkdir):
        """Test handling of invalid resolution format."""
        config = Config()
        # Should fall back to default
        assert config.camera.main_resolution == (1920, 1080)
    
    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat',
        'CAMERA_MOTION_RESOLUTION': '1920x1080',
        'CAMERA_MAIN_RESOLUTION': '640x480'
    })
    @patch('pathlib.Path.mkdir')
    def test_resolution_consistency_validation(self, mock_mkdir):
        """Test that motion resolution cannot be larger than main resolution."""
        with pytest.raises(ConfigurationError, match="Motion detection resolution cannot be larger"):
            Config()
    
    def test_create_test_config(self):
        """Test test configuration creation."""
        config = Config.create_test_config()
        assert config.telegram_token == 'test_token'
        assert config.telegram_chat_id == 'test_chat'
    
    def test_create_test_config_with_overrides(self):
        """Test test configuration with overrides."""
        config = Config.create_test_config(
            MOTION_THRESHOLD='5000'
        )
        assert config.motion.motion_threshold == 5000
    
    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat'
    })
    @patch('pathlib.Path.mkdir')
    def test_get_summary(self, mock_mkdir):
        """Test configuration summary."""
        config = Config()
        summary = config.get_summary()
        
        assert 'camera' in summary
        assert 'motion' in summary
        assert 'performance' in summary
        assert 'storage' in summary
        
        assert summary['camera']['main_resolution'] == (1920, 1080)
        assert summary['motion']['threshold'] == 2000


if __name__ == '__main__':
    pytest.main([__file__])