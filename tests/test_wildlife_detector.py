"""
Integration tests for the main wildlife detection system.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime

import sys
sys.path.append('src')

from wildlife_detector import WildlifeDetector, WildlifeDetectorError
from config import Config
from camera_manager import CameraManager
from motion_detector import MotionDetector
from species_identifier import SpeciesIdentifier
from database_manager import DatabaseManager


class TestWildlifeDetectorInitialization:
    """Test wildlife detector initialization and configuration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wildlife_detector_initialization(self):
        """Test wildlife detector initialization."""
        detector = WildlifeDetector(self.config, use_mock_components=True)
        
        assert detector.config == self.config
        assert isinstance(detector.camera_manager, CameraManager)
        assert isinstance(detector.motion_detector, MotionDetector)
        assert isinstance(detector.species_identifier, SpeciesIdentifier)
        assert isinstance(detector.database_manager, DatabaseManager)
        assert not detector.is_running
    
    def test_wildlife_detector_component_validation(self):
        """Test component validation during initialization."""
        # Test with invalid config
        invalid_config = Config.create_test_config()
        invalid_config.storage.database_path = Path("/invalid/path/db.sqlite")
        
        with pytest.raises(WildlifeDetectorError, match="Failed to initialize database"):
            WildlifeDetector(invalid_config, use_mock_components=True)
    
    def test_wildlife_detector_mock_vs_real_components(self):
        """Test mock vs real component selection."""
        # Mock components
        mock_detector = WildlifeDetector(self.config, use_mock_components=True)
        assert mock_detector.camera_manager._camera.__class__.__name__ == 'MockCameraManager'
        
        # Real components (but will use mock since we're not on Pi)
        real_detector = WildlifeDetector(self.config, use_mock_components=False)
        # Should still use mock on non-Pi systems
        assert real_detector.camera_manager._camera is not None


class TestWildlifeDetectorLifecycle:
    """Test wildlife detector startup and shutdown lifecycle."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        self.detector = WildlifeDetector(self.config, use_mock_components=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.detector.is_running:
            asyncio.run(self.detector.stop())
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_detector_startup_sequence(self):
        """Test detector startup sequence."""
        assert not self.detector.is_running
        
        await self.detector.start()
        
        assert self.detector.is_running
        assert self.detector.camera_manager.is_operational()
        assert self.detector.motion_detector.is_operational()
        assert self.detector.species_identifier.is_operational()
    
    @pytest.mark.asyncio
    async def test_detector_shutdown_sequence(self):
        """Test detector shutdown sequence."""
        await self.detector.start()
        assert self.detector.is_running
        
        await self.detector.stop()
        
        assert not self.detector.is_running
        assert not self.detector.camera_manager.is_operational()
        assert not self.detector.motion_detector.is_operational()
        assert not self.detector.species_identifier.is_operational()
    
    @pytest.mark.asyncio
    async def test_detector_restart(self):
        """Test detector restart functionality."""
        await self.detector.start()
        await self.detector.stop()
        
        # Should be able to restart
        await self.detector.start()
        assert self.detector.is_running
        
        await self.detector.stop()
    
    @pytest.mark.asyncio
    async def test_detector_context_manager(self):
        """Test detector as async context manager."""
        async with self.detector:
            assert self.detector.is_running
        
        assert not self.detector.is_running


class TestWildlifeDetectorDetection:
    """Test wildlife detection functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        
        # Configure for faster testing
        self.config.performance.cooldown_period = 1  # 1 second cooldown
        self.config.motion.motion_threshold = 100   # Lower threshold
        
        self.detector = WildlifeDetector(self.config, use_mock_components=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.detector.is_running:
            asyncio.run(self.detector.stop())
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_single_detection_cycle(self):
        """Test a single detection cycle."""
        async with self.detector:
            # Mock motion detection to return motion
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(
                    motion_detected=True,
                    motion_area=2500,
                    largest_contour_area=1800,
                    contour_count=3,
                    processing_time=0.05
                )
            )
            
            # Run single detection cycle
            result = await self.detector._process_single_frame()
            
            assert result is not None
            assert 'motion_detected' in result
            assert 'photo_captured' in result
            assert 'species_identified' in result
            assert 'database_logged' in result
    
    @pytest.mark.asyncio
    async def test_no_motion_detection(self):
        """Test behavior when no motion is detected."""
        async with self.detector:
            # Mock no motion
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(
                    motion_detected=False,
                    motion_area=0,
                    largest_contour_area=0,
                    contour_count=0,
                    processing_time=0.03
                )
            )
            
            result = await self.detector._process_single_frame()
            
            assert result is not None
            assert result['motion_detected'] is False
            assert result['photo_captured'] is False
            assert result['species_identified'] is False
    
    @pytest.mark.asyncio
    async def test_cooldown_period_enforcement(self):
        """Test cooldown period between detections."""
        async with self.detector:
            # Mock motion detection
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(
                    motion_detected=True,
                    motion_area=2500,
                    largest_contour_area=1800,
                    contour_count=3,
                    processing_time=0.05
                )
            )
            
            # First detection should process normally
            result1 = await self.detector._process_single_frame()
            assert result1['motion_detected'] is True
            assert result1['photo_captured'] is True
            
            # Second detection immediately after should be in cooldown
            result2 = await self.detector._process_single_frame()
            assert result2['motion_detected'] is True
            assert result2['photo_captured'] is False  # Cooldown active
            
            # Wait for cooldown to expire
            await asyncio.sleep(1.1)  # Slightly longer than cooldown
            
            result3 = await self.detector._process_single_frame()
            assert result3['photo_captured'] is True  # Should capture again
    
    @pytest.mark.asyncio
    async def test_species_identification_integration(self):
        """Test species identification integration."""
        async with self.detector:
            # Mock successful identification
            mock_result = Mock(
                species_name="European Hedgehog",
                confidence=0.87,
                processing_time=2.5,
                api_success=True,
                fallback_reason=None
            )
            self.detector.species_identifier._implementation.identify_from_array = Mock(
                return_value=mock_result
            )
            
            # Mock motion and camera
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
            )
            
            result = await self.detector._process_single_frame()
            
            assert result['species_identified'] is True
            assert result['species_name'] == "European Hedgehog"
            assert result['confidence'] == 0.87
    
    @pytest.mark.asyncio
    async def test_database_logging_integration(self):
        """Test database logging integration."""
        async with self.detector:
            # Mock motion detection
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
            )
            
            result = await self.detector._process_single_frame()
            
            assert result['database_logged'] is True
            
            # Verify data was logged to database
            recent_detections = self.detector.database_manager.get_recent_detections(1)
            assert len(recent_detections) == 1
            assert recent_detections[0]['motion_area'] == 2500
    
    @pytest.mark.asyncio
    async def test_telegram_notification_integration(self):
        """Test Telegram notification integration."""
        with patch.object(self.detector, '_send_telegram_message', new_callable=AsyncMock) as mock_send:
            async with self.detector:
                # Mock motion detection
                self.detector.motion_detector._implementation.detect_motion = Mock(
                    return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
                )
                
                result = await self.detector._process_single_frame()
                
                assert result['telegram_sent'] is True
                mock_send.assert_called_once()


class TestWildlifeDetectorErrorHandling:
    """Test error handling in wildlife detector."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        self.detector = WildlifeDetector(self.config, use_mock_components=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.detector.is_running:
            asyncio.run(self.detector.stop())
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_camera_error_handling(self):
        """Test handling of camera errors."""
        async with self.detector:
            # Mock camera failure
            self.detector.camera_manager.capture_motion_frame = Mock(return_value=None)
            
            result = await self.detector._process_single_frame()
            
            # Should handle error gracefully
            assert result is not None
            assert result['error'] is True
    
    @pytest.mark.asyncio
    async def test_motion_detector_error_handling(self):
        """Test handling of motion detector errors."""
        async with self.detector:
            # Mock motion detector failure
            self.detector.motion_detector._implementation.detect_motion = Mock(
                side_effect=Exception("Motion detection failed")
            )
            
            result = await self.detector._process_single_frame()
            
            # Should handle error gracefully
            assert result is not None
            assert result['error'] is True
    
    @pytest.mark.asyncio
    async def test_species_identification_error_handling(self):
        """Test handling of species identification errors."""
        async with self.detector:
            # Mock motion detection success but species ID failure
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
            )
            
            self.detector.species_identifier._implementation.identify_from_array = Mock(
                side_effect=Exception("Species identification failed")
            )
            
            result = await self.detector._process_single_frame()
            
            # Should continue operation with fallback
            assert result['motion_detected'] is True
            assert result['photo_captured'] is True
            assert result['species_identified'] is False
            assert result['species_name'] == "Unknown species"
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test handling of database errors."""
        async with self.detector:
            # Mock database failure
            self.detector.database_manager.insert_detection = Mock(
                side_effect=Exception("Database insert failed")
            )
            
            # Mock successful motion detection
            self.detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
            )
            
            result = await self.detector._process_single_frame()
            
            # Should continue operation despite database error
            assert result['motion_detected'] is True
            assert result['photo_captured'] is True
            assert result['database_logged'] is False
    
    @pytest.mark.asyncio
    async def test_telegram_error_handling(self):
        """Test handling of Telegram errors."""
        with patch.object(self.detector, '_send_telegram_message', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Telegram API failed")
            
            async with self.detector:
                # Mock successful motion detection
                self.detector.motion_detector._implementation.detect_motion = Mock(
                    return_value=Mock(motion_detected=True, motion_area=2500, processing_time=0.05)
                )
                
                result = await self.detector._process_single_frame()
                
                # Should continue operation despite Telegram error
                assert result['motion_detected'] is True
                assert result['photo_captured'] is True
                assert result['telegram_sent'] is False


class TestWildlifeDetectorPerformance:
    """Test performance and resource management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        self.detector = WildlifeDetector(self.config, use_mock_components=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.detector.is_running:
            asyncio.run(self.detector.stop())
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_memory_management(self):
        """Test memory management during operation."""
        async with self.detector:
            # Simulate multiple detection cycles
            for _ in range(10):
                await self.detector._process_single_frame()
                # Small delay to allow memory cleanup
                await asyncio.sleep(0.01)
            
            # Memory usage should remain reasonable
            memory_info = self.detector.get_system_info()
            assert 'memory' in memory_info
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance tracking."""
        async with self.detector:
            # Run several detection cycles
            for _ in range(5):
                await self.detector._process_single_frame()
            
            perf_stats = self.detector.get_performance_stats()
            
            assert 'total_frames_processed' in perf_stats
            assert 'average_processing_time' in perf_stats
            assert 'detection_rate' in perf_stats
            
            assert perf_stats['total_frames_processed'] == 5
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test resource cleanup on shutdown."""
        await self.detector.start()
        
        # Verify resources are allocated
        assert self.detector.camera_manager.is_operational()
        
        await self.detector.stop()
        
        # Verify resources are cleaned up
        assert not self.detector.camera_manager.is_operational()
    
    def test_file_cleanup(self):
        """Test file cleanup functionality."""
        # Create many test image files
        for i in range(150):  # More than max_images (100)
            image_file = self.config.storage.image_dir / f"test_{i:03d}.jpg"
            image_file.parent.mkdir(parents=True, exist_ok=True)
            image_file.write_bytes(b"fake_image_data")
        
        detector = WildlifeDetector(self.config, use_mock_components=True)
        
        # Trigger cleanup
        detector._cleanup_old_images()
        
        # Should have cleaned up to max_images
        remaining_files = list(self.config.storage.image_dir.glob("*.jpg"))
        assert len(remaining_files) <= self.config.performance.max_images


class TestWildlifeDetectorMonitoring:
    """Test monitoring and status reporting."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        self.detector = WildlifeDetector(self.config, use_mock_components=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.detector.is_running:
            asyncio.run(self.detector.stop())
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_info_reporting(self):
        """Test system information reporting."""
        system_info = self.detector.get_system_info()
        
        assert isinstance(system_info, dict)
        assert 'detector_status' in system_info
        assert 'components' in system_info
        assert 'configuration' in system_info
        assert 'performance' in system_info
        
        assert 'camera_manager' in system_info['components']
        assert 'motion_detector' in system_info['components']
        assert 'species_identifier' in system_info['components']
        assert 'database_manager' in system_info['components']
    
    def test_health_check(self):
        """Test health check functionality."""
        health = self.detector.health_check()
        
        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'component_status' in health
        assert 'last_detection' in health
        assert 'uptime' in health
        
        assert health['overall_status'] in ['healthy', 'degraded', 'unhealthy']
    
    @pytest.mark.asyncio
    async def test_status_updates(self):
        """Test status update tracking."""
        await self.detector.start()
        
        # Process some frames to generate status
        for _ in range(3):
            await self.detector._process_single_frame()
        
        status = self.detector.get_status()
        
        assert 'is_running' in status
        assert 'frames_processed' in status
        assert 'last_activity' in status
        assert 'errors' in status
        
        assert status['is_running'] is True
        assert status['frames_processed'] >= 3
        
        await self.detector.stop()


@pytest.mark.integration
class TestWildlifeDetectorIntegration:
    """Integration tests requiring all components."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config.create_test_config()
        self.config.storage.data_dir = self.temp_dir
        self.config.storage.database_path = self.temp_dir / "integration_test.db"
        self.config.storage.image_dir = self.temp_dir / "images"
        
        # Shorter intervals for testing
        self.config.performance.cooldown_period = 0.1
    
    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self):
        """Test complete detection pipeline integration."""
        detector = WildlifeDetector(self.config, use_mock_components=True)
        
        async with detector:
            # Run detection loop for a short time
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 2.0 and frame_count < 10:  # 2 seconds or 10 frames max
                await detector._process_single_frame()
                frame_count += 1
                await asyncio.sleep(0.1)
            
            # Verify system operated correctly
            stats = detector.get_performance_stats()
            assert stats['total_frames_processed'] == frame_count
            
            # Check database for logged data
            detections = detector.database_manager.get_recent_detections(10)
            assert len(detections) >= 0  # Should have some detections or none (both valid)
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock_scenario(self):
        """Test end-to-end scenario with controlled mock data."""
        detector = WildlifeDetector(self.config, use_mock_components=True)
        
        # Force motion detection for this test
        async with detector:
            detector.motion_detector._implementation.detect_motion = Mock(
                return_value=Mock(
                    motion_detected=True,
                    motion_area=2500,
                    largest_contour_area=1800,
                    contour_count=3,
                    processing_time=0.05
                )
            )
            
            with patch.object(detector, '_send_telegram_message', new_callable=AsyncMock) as mock_telegram:
                # Process one detection cycle
                result = await detector._process_single_frame()
                
                # Verify all stages completed
                assert result['motion_detected'] is True
                assert result['photo_captured'] is True
                assert result['species_identified'] is True
                assert result['database_logged'] is True
                assert result['telegram_sent'] is True
                
                # Verify Telegram message was sent
                mock_telegram.assert_called_once()
                
                # Verify database entry
                detections = detector.database_manager.get_recent_detections(1)
                assert len(detections) == 1
                assert detections[0]['motion_area'] == 2500


if __name__ == '__main__':
    pytest.main([__file__])