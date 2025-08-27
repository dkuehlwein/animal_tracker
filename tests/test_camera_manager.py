"""
Unit tests for camera management system.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

import sys
sys.path.append('src')

from camera_manager import (
    CameraManager, MockCameraManager, PiCameraManager,
    CameraError, CameraInitializationError, CameraOperationError,
    ResourceManager
)
from config import Config


class TestResourceManager:
    """Test resource management functionality."""
    
    def test_resource_registration(self):
        """Test frame resource registration and cleanup."""
        rm = ResourceManager()
        
        # Test registration
        rm.register_frame("frame_1")
        rm.register_frame("frame_2")
        assert rm.get_active_count() == 2
        
        # Test unregistration
        rm.unregister_frame("frame_1")
        assert rm.get_active_count() == 1
        
        # Test cleanup all
        rm.cleanup_all()
        assert rm.get_active_count() == 0
    
    def test_thread_safety(self):
        """Test thread safety of resource manager."""
        import threading
        
        rm = ResourceManager()
        errors = []
        
        def register_frames(start_id):
            try:
                for i in range(100):
                    rm.register_frame(f"frame_{start_id}_{i}")
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_frames, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have no errors and correct count
        assert len(errors) == 0
        assert rm.get_active_count() == 500


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
        assert frame.shape == (1080, 1920, 3)  # height, width, channels
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


class TestPiCameraManager:
    """Test Pi camera implementation (mocked)."""
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_initialization_success(self, mock_picamera2):
        """Test successful Pi camera initialization."""
        # Mock Picamera2
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        
        camera.start()
        
        # Verify camera was configured and started
        assert mock_camera_instance.create_preview_configuration.called
        assert mock_camera_instance.configure.called
        assert mock_camera_instance.start.called
        assert camera.is_available()
    
    @patch('camera_manager.Picamera2', side_effect=ImportError("No module named 'picamera2'"))
    def test_pi_camera_import_error(self, mock_picamera2):
        """Test handling of missing Picamera2 library."""
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        
        with pytest.raises(CameraInitializationError, match="Picamera2 library not available"):
            camera.start()
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_configuration_error(self, mock_picamera2):
        """Test handling of camera configuration errors."""
        mock_camera_instance = Mock()
        mock_camera_instance.create_preview_configuration.side_effect = Exception("Config failed")
        mock_picamera2.return_value = mock_camera_instance
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        
        with pytest.raises(CameraInitializationError, match="Failed to configure camera streams"):
            camera.start()
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_retry_logic(self, mock_picamera2):
        """Test camera initialization retry logic."""
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        
        # Fail first two attempts, succeed on third
        mock_camera_instance.start.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            None  # Success
        ]
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        
        # Should succeed after retries
        camera.start()
        assert camera.is_available()
        
        # Should have been called 3 times
        assert mock_camera_instance.start.call_count == 3
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_max_retries_exceeded(self, mock_picamera2):
        """Test behavior when max retries are exceeded."""
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        mock_camera_instance.start.side_effect = Exception("Persistent failure")
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        
        with pytest.raises(CameraInitializationError, match="Failed to initialize camera after 3 attempts"):
            camera.start()
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_motion_frame_capture(self, mock_picamera2):
        """Test Pi camera motion frame capture."""
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        
        # Mock YUV frame data
        mock_yuv_frame = np.random.randint(0, 255, (720, 640), dtype=np.uint8)  # Larger than needed
        mock_camera_instance.capture_array.return_value = mock_yuv_frame
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        camera.start()
        
        frame = camera.capture_motion_frame()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640)  # Should be cropped to motion detection size
        assert frame.dtype == np.uint8
        
        # Verify capture_array was called with "lores"
        mock_camera_instance.capture_array.assert_called_with("lores")
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_capture_error_handling(self, mock_picamera2):
        """Test error handling during frame capture."""
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        mock_camera_instance.capture_array.side_effect = Exception("Capture failed")
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        camera.start()
        
        frame = camera.capture_motion_frame()
        assert frame is None
        
        stats = camera.get_stats()
        assert stats['error_count'] > 0
    
    @patch('camera_manager.Picamera2')
    def test_pi_camera_error_recovery(self, mock_picamera2):
        """Test automatic error recovery."""
        mock_camera_instance = Mock()
        mock_picamera2.return_value = mock_camera_instance
        
        config = Config.create_test_config()
        camera = PiCameraManager(config)
        camera.start()
        
        # Simulate many errors
        for _ in range(12):  # More than recovery threshold
            camera._handle_capture_error("Test error")
        
        # Should have attempted restart
        assert mock_camera_instance.stop.call_count >= 1


class TestCameraManager:
    """Test high-level camera manager."""
    
    def test_camera_manager_with_mock(self):
        """Test camera manager with mock camera."""
        config = Config.create_test_config()
        manager = CameraManager(config, use_mock=True)
        
        # Test context manager
        with manager:
            assert manager.is_operational()
            
            # Test motion frame capture
            frame = manager.capture_motion_frame()
            assert frame is not None
        
        # Should be stopped after context
        assert not manager.is_operational()
    
    def test_camera_manager_capture_and_save(self, tmp_path):
        """Test photo capture and save functionality."""
        config = Config.create_test_config()
        # Override image directory for test
        config.storage.image_dir = tmp_path
        
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
        assert system_info['configuration']['main_resolution'] == (1920, 1080)
    
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


if __name__ == '__main__':
    pytest.main([__file__])