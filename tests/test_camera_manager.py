"""
Unit tests for camera management system.
"""

import sys
from types import ModuleType

import pytest
import numpy as np
from unittest.mock import Mock

sys.path.append('src')

from camera_manager import CameraManager, MockCameraManager, PiCameraManager
from config import Config


def _install_fake_picamera2(monkeypatch, mock_camera_instance):
    """Inject a fake `picamera2` module into sys.modules.

    `PiCameraManager._initialize_camera` does `from picamera2 import Picamera2`
    locally (guarded by try/except ImportError) so it works whether or not the
    real hardware library is installed. Overriding the sys.modules entry lets
    the local import resolve to our mock without touching real hardware.
    """
    fake_module = ModuleType('picamera2')
    fake_module.Picamera2 = Mock(return_value=mock_camera_instance)
    monkeypatch.setitem(sys.modules, 'picamera2', fake_module)


def _install_fake_libcamera(monkeypatch):
    """Inject a fake `libcamera` module exposing `controls.AeExposureModeEnum`.

    Mirrors the Picamera2 mocking approach above for the guarded
    `from libcamera import controls` import used by the AE-exposure-mode fix.
    """
    fake_controls = Mock()
    fake_controls.AeExposureModeEnum.Normal = "ENUM_NORMAL"
    fake_controls.AeExposureModeEnum.Short = "ENUM_SHORT"
    fake_controls.AeExposureModeEnum.Long = "ENUM_LONG"

    fake_module = ModuleType('libcamera')
    fake_module.controls = fake_controls
    monkeypatch.setitem(sys.modules, 'libcamera', fake_module)



class TestMockCameraManager:
    """Test mock camera implementation."""
    
    def test_mock_camera_lifecycle(self):
        """Test mock camera start/stop lifecycle."""
        config = Config.create_test_config()
        camera = MockCameraManager(config)
        
        # Initially not running
        assert not camera.is_available()
        
        # Start camera
        camera.start()
        assert camera.is_available()
        
        # Stop camera
        camera.stop()
        assert not camera.is_available()
    
    def test_mock_motion_frame_capture(self):
        """Test mock motion frame capture."""
        config = Config.create_test_config()
        camera = MockCameraManager(config)
        camera.start()
        
        frame = camera.capture_motion_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640)  # height, width
        assert frame.dtype == np.uint8
        
        camera.stop()
        
        # Should return None when stopped
        frame = camera.capture_motion_frame()
        assert frame is None
    
    def test_mock_high_res_frame_capture(self):
        """Test mock high resolution frame capture."""
        config = Config.create_test_config()
        camera = MockCameraManager(config)
        camera.start()
        
        frame = camera.capture_high_res_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1520, 2028, 3)  # height, width, channels
        assert frame.dtype == np.uint8
    
    def test_mock_save_frame_to_file(self, tmp_path):
        """Test mock frame saving."""
        config = Config.create_test_config()
        camera = MockCameraManager(config)
        
        # Create mock frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        file_path = tmp_path / "test_frame.jpg"
        
        success = camera.save_frame_to_file(frame, file_path)
        assert success
        assert file_path.exists()
    
    def test_mock_stats(self):
        """Test mock camera statistics."""
        config = Config.create_test_config()
        camera = MockCameraManager(config)
        
        stats = camera.get_stats()
        assert stats['is_running'] is False
        assert stats['error_count'] == 0
        assert stats['active_frames'] == 0
        assert stats['last_error_time'] == 0


# PiCameraManager tests removed - these were pure mocking tests that don't test actual behavior
# The actual camera behavior is tested through integration tests with real hardware


class TestCameraManager:
    """Test high-level camera manager."""
    
    def test_camera_manager_with_mock(self):
        """Test camera manager with mock camera."""
        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)

        manager.start()
        try:
            assert manager.is_operational()

            # Test motion frame capture
            frame = manager.capture_motion_frame()
            assert frame is not None
        finally:
            manager.stop()

        # Should be stopped after context
        assert not manager.is_operational()
    
    def test_camera_manager_capture_and_save(self, tmp_path):
        """Test photo capture and save functionality."""
        # Create test directory
        tmp_path.mkdir(parents=True, exist_ok=True)

        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)
        manager.start()

        try:
            photo_path = manager.capture_and_save_photo()

            assert photo_path is not None
            assert photo_path.exists()
            assert photo_path.suffix == '.jpg'
            assert config.storage.image_prefix in photo_path.name
        finally:
            manager.stop()
    
    def test_camera_manager_system_info(self):
        """Test system information retrieval."""
        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)
        
        system_info = manager.get_system_info()
        
        assert 'camera_type' in system_info
        assert 'configuration' in system_info
        assert 'stats' in system_info
        assert 'storage' in system_info
        
        assert system_info['camera_type'] == 'MockCameraManager'
        assert system_info['configuration']['main_resolution'] == (2028, 1520)
    
    def test_camera_manager_session_context(self):
        """Test camera session context manager."""
        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)
        
        with manager.camera_session() as camera:
            assert camera.is_available()
            frame = camera.capture_motion_frame()
            assert frame is not None
        
        # Camera should be stopped after session
        assert not manager.is_operational()
    
    def test_camera_manager_capture_failure_handling(self):
        """Test handling of capture failures."""
        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)
        
        # Mock the camera to return None for high res capture
        manager._camera.capture_high_res_frame = Mock(return_value=None)
        
        manager.start()
        try:
            photo_path = manager.capture_and_save_photo()
            assert photo_path is None
        finally:
            manager.stop()


class TestPiCameraManagerAeExposureMode:
    """Dusk-exposure fix: bias AE toward short exposure via AeExposureMode.

    Uses PiCameraManager with picamera2/libcamera mocked out (no hardware
    required) to assert the controls dict passed to `set_controls()`.
    """

    def test_auto_exposure_mode_sets_mapped_enum(self, monkeypatch):
        """Default (short) maps to the libcamera enum in auto-exposure mode."""
        mock_camera_instance = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        _install_fake_libcamera(monkeypatch)

        config = Config.create_test_config()
        assert config.camera.exposure_time is None
        assert config.camera.analogue_gain is None
        camera = PiCameraManager(config)

        camera.start()

        applied_controls = mock_camera_instance.set_controls.call_args[0][0]
        assert applied_controls["AeExposureMode"] == "ENUM_SHORT"

    def test_ae_exposure_mode_env_override_maps_to_long(self, monkeypatch):
        """CAMERA_AE_EXPOSURE_MODE=long maps to the Long enum member."""
        monkeypatch.setenv("CAMERA_AE_EXPOSURE_MODE", "long")
        mock_camera_instance = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        _install_fake_libcamera(monkeypatch)

        config = Config.create_test_config()
        camera = PiCameraManager(config)

        camera.start()

        applied_controls = mock_camera_instance.set_controls.call_args[0][0]
        assert applied_controls["AeExposureMode"] == "ENUM_LONG"

    def test_manual_exposure_mode_omits_ae_exposure_mode_control(self, monkeypatch):
        """Manual exposure branch (:158-161) is untouched: no AeExposureMode key."""
        monkeypatch.setenv("CAMERA_EXPOSURE_TIME", "2000")
        monkeypatch.setenv("CAMERA_ANALOGUE_GAIN", "2.5")
        mock_camera_instance = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        _install_fake_libcamera(monkeypatch)

        config = Config.create_test_config()
        camera = PiCameraManager(config)

        camera.start()

        applied_controls = mock_camera_instance.set_controls.call_args[0][0]
        assert "AeExposureMode" not in applied_controls
        # Manual exposure controls are still set exactly as before.
        assert applied_controls["AeEnable"] is False
        assert applied_controls["ExposureTime"] == 2000
        assert applied_controls["AnalogueGain"] == 2.5

    def test_camera_hardware_still_started_in_both_modes(self, monkeypatch):
        """Regression guard: hardware start() must run in auto AND manual mode."""
        mock_camera_instance = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        _install_fake_libcamera(monkeypatch)

        config = Config.create_test_config()
        camera = PiCameraManager(config)
        camera.start()
        assert mock_camera_instance.start.called
        assert camera.is_available()

        monkeypatch.setenv("CAMERA_EXPOSURE_TIME", "2000")
        monkeypatch.setenv("CAMERA_ANALOGUE_GAIN", "2.5")
        mock_camera_instance2 = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance2)
        config2 = Config.create_test_config()
        camera2 = PiCameraManager(config2)
        camera2.start()
        assert mock_camera_instance2.start.called
        assert camera2.is_available()

    def test_libcamera_unavailable_degrades_gracefully(self, monkeypatch):
        """libcamera not importable: warn and continue without AeExposureMode."""
        mock_camera_instance = Mock()
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        monkeypatch.delitem(sys.modules, 'libcamera', raising=False)
        monkeypatch.setitem(sys.modules, 'libcamera', None)  # force ImportError

        config = Config.create_test_config()
        camera = PiCameraManager(config)

        camera.start()  # must not raise

        assert camera.is_available()
        applied_controls = mock_camera_instance.set_controls.call_args[0][0]
        assert "AeExposureMode" not in applied_controls
        # Other controls still applied.
        assert "FrameDurationLimits" in applied_controls

    def test_set_controls_exception_does_not_propagate(self, monkeypatch):
        """A rejected control (e.g. unsupported enum/sensor) must not crash startup."""
        mock_camera_instance = Mock()
        mock_camera_instance.set_controls.side_effect = Exception("control rejected")
        _install_fake_picamera2(monkeypatch, mock_camera_instance)
        _install_fake_libcamera(monkeypatch)

        config = Config.create_test_config()
        camera = PiCameraManager(config)

        camera.start()  # must not raise

        assert camera.is_available()


if __name__ == '__main__':
    pytest.main([__file__])