"""
Unit tests for utility functions and helpers.
"""

import pytest
import tempfile
import shutil
import gc
import psutil
import time
from pathlib import Path
from unittest.mock import patch, Mock, mock_open

import sys
sys.path.append('src')

from utils import (
    MemoryManager, FileManager, ImageUtils, TelegramFormatter,
    SystemMonitor, PerformanceTracker, UtilsError
)
from config import Config


class TestMemoryManager:
    """Test memory management utilities."""
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        usage = MemoryManager.get_memory_usage()
        
        assert isinstance(usage, dict)
        assert 'total_mb' in usage
        assert 'available_mb' in usage
        assert 'used_mb' in usage
        assert 'percent_used' in usage
        
        assert usage['total_mb'] > 0
        assert usage['available_mb'] > 0
        assert 0 <= usage['percent_used'] <= 100
    
    def test_is_memory_pressure(self):
        """Test memory pressure detection."""
        # Test with low threshold (should not be under pressure)
        assert not MemoryManager.is_memory_pressure(threshold=0.1)
        
        # Test with high threshold (might be under pressure)
        high_pressure = MemoryManager.is_memory_pressure(threshold=0.95)
        assert isinstance(high_pressure, bool)
    
    def test_cleanup_variables(self):
        """Test variable cleanup functionality."""
        # Create some large variables
        large_data = [list(range(1000)) for _ in range(100)]
        other_data = "test_string"
        
        # Cleanup specific variables
        MemoryManager.cleanup_variables(large_data, other_data)
        
        # Variables should be cleaned up (set to None)
        # Note: This test verifies the function runs without error
        # The actual cleanup effectiveness depends on Python's GC
    
    def test_force_garbage_collection(self):
        """Test garbage collection forcing."""
        # Create some objects to be collected
        temp_objects = [object() for _ in range(1000)]
        temp_objects.clear()
        
        # Force collection
        collected = MemoryManager.force_garbage_collection()
        
        assert isinstance(collected, int)
        assert collected >= 0
    
    def test_get_process_memory(self):
        """Test process-specific memory information."""
        mem_info = MemoryManager.get_process_memory()
        
        assert isinstance(mem_info, dict)
        assert 'rss_mb' in mem_info  # Resident Set Size
        assert 'vms_mb' in mem_info  # Virtual Memory Size
        assert 'percent' in mem_info
        
        assert mem_info['rss_mb'] > 0
        assert mem_info['vms_mb'] > 0
        assert mem_info['percent'] >= 0


class TestFileManager:
    """Test file management utilities."""
    
    def setup_method(self):
        """Set up test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        test_dir = self.temp_dir / "subdir" / "nested"
        
        FileManager.ensure_directory(test_dir)
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_cleanup_old_files(self):
        """Test old file cleanup."""
        # Create test files with different ages
        old_file = self.temp_dir / "old_file.txt"
        recent_file = self.temp_dir / "recent_file.txt"
        
        old_file.write_text("old content")
        recent_file.write_text("recent content")
        
        # Make old file appear old by modifying timestamp
        old_time = time.time() - (25 * 60 * 60)  # 25 hours ago
        old_file.touch(times=(old_time, old_time))
        
        # Clean up files older than 24 hours
        removed_count = FileManager.cleanup_old_files(self.temp_dir, hours=24)
        
        assert removed_count == 1
        assert not old_file.exists()
        assert recent_file.exists()
    
    def test_cleanup_old_files_with_pattern(self):
        """Test old file cleanup with pattern matching."""
        # Create various files
        files_to_create = [
            "image_001.jpg",
            "image_002.png", 
            "log_file.txt",
            "data.json"
        ]
        
        for filename in files_to_create:
            file_path = self.temp_dir / filename
            file_path.write_text("content")
            # Make all files old
            old_time = time.time() - (25 * 60 * 60)
            file_path.touch(times=(old_time, old_time))
        
        # Clean up only image files
        removed_count = FileManager.cleanup_old_files(
            self.temp_dir, hours=24, pattern="image_*"
        )
        
        assert removed_count == 2
        assert not (self.temp_dir / "image_001.jpg").exists()
        assert not (self.temp_dir / "image_002.png").exists()
        assert (self.temp_dir / "log_file.txt").exists()
        assert (self.temp_dir / "data.json").exists()
    
    def test_get_directory_size(self):
        """Test directory size calculation."""
        # Create files with known sizes
        (self.temp_dir / "file1.txt").write_text("a" * 1000)  # 1000 bytes
        (self.temp_dir / "file2.txt").write_text("b" * 2000)  # 2000 bytes
        
        # Create subdirectory with more files
        subdir = self.temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("c" * 500)  # 500 bytes
        
        total_size = FileManager.get_directory_size(self.temp_dir)
        
        # Should be approximately 3500 bytes (allowing for filesystem overhead)
        assert total_size >= 3500
        assert total_size <= 4000  # Some buffer for filesystem metadata
    
    def test_safe_file_operation(self):
        """Test safe file operations with atomic writes."""
        target_file = self.temp_dir / "safe_write_test.txt"
        test_content = "This is test content"
        
        # Test successful write
        success = FileManager.safe_write_file(target_file, test_content)
        
        assert success
        assert target_file.exists()
        assert target_file.read_text() == test_content
    
    def test_safe_file_operation_failure(self):
        """Test safe file operation failure handling."""
        # Try to write to invalid location
        invalid_path = Path("/invalid/path/file.txt")
        
        success = FileManager.safe_write_file(invalid_path, "content")
        
        assert not success
    
    def test_get_file_info(self):
        """Test file information retrieval."""
        test_file = self.temp_dir / "info_test.txt"
        test_content = "Test file content"
        test_file.write_text(test_content)
        
        file_info = FileManager.get_file_info(test_file)
        
        assert isinstance(file_info, dict)
        assert 'size_bytes' in file_info
        assert 'created' in file_info
        assert 'modified' in file_info
        assert 'is_file' in file_info
        
        assert file_info['size_bytes'] == len(test_content)
        assert file_info['is_file'] is True


class TestImageUtils:
    """Test image utility functions."""
    
    def test_generate_filename(self):
        """Test image filename generation."""
        filename = ImageUtils.generate_filename("capture_", "jpg")
        
        assert filename.startswith("capture_")
        assert filename.endswith(".jpg")
        assert len(filename.split("_")[1].split(".")[0]) == 14  # Timestamp format
    
    def test_generate_filename_with_suffix(self):
        """Test filename generation with custom suffix."""
        filename = ImageUtils.generate_filename("img_", "png", suffix="_motion")
        
        assert filename.startswith("img_")
        assert "_motion" in filename
        assert filename.endswith(".png")
    
    def test_validate_image_format(self):
        """Test image format validation."""
        assert ImageUtils.validate_image_format("test.jpg")
        assert ImageUtils.validate_image_format("test.jpeg")
        assert ImageUtils.validate_image_format("test.png")
        assert ImageUtils.validate_image_format("test.bmp")
        
        assert not ImageUtils.validate_image_format("test.txt")
        assert not ImageUtils.validate_image_format("test.pdf")
        assert not ImageUtils.validate_image_format("test")
    
    def test_get_image_dimensions_info(self):
        """Test image dimension information."""
        # Test various resolutions
        test_cases = [
            ((1920, 1080), "Full HD"),
            ((1280, 720), "HD"),
            ((640, 480), "VGA"),
            ((320, 240), "QVGA"),
        ]
        
        for resolution, expected_name in test_cases:
            info = ImageUtils.get_image_dimensions_info(resolution)
            assert info['resolution'] == resolution
            assert info['aspect_ratio'] == round(resolution[0] / resolution[1], 2)
            assert expected_name.lower() in info['common_name'].lower()
    
    def test_calculate_resize_dimensions(self):
        """Test image resize calculation."""
        # Test downscaling
        new_dims = ImageUtils.calculate_resize_dimensions(
            (1920, 1080), max_width=640
        )
        assert new_dims[0] == 640
        assert new_dims[1] == 360  # Maintain aspect ratio
        
        # Test upscaling prevention
        new_dims = ImageUtils.calculate_resize_dimensions(
            (640, 480), max_width=1920, allow_upscale=False
        )
        assert new_dims == (640, 480)  # Should remain unchanged
    
    def test_estimate_compression_ratio(self):
        """Test compression ratio estimation."""
        # Test different image types
        jpeg_ratio = ImageUtils.estimate_compression_ratio((1920, 1080), "jpg")
        png_ratio = ImageUtils.estimate_compression_ratio((1920, 1080), "png")
        
        assert isinstance(jpeg_ratio, float)
        assert isinstance(png_ratio, float)
        assert jpeg_ratio > png_ratio  # JPEG should compress more


class TestTelegramFormatter:
    """Test Telegram message formatting utilities."""
    
    def test_format_detection_message(self):
        """Test detection message formatting."""
        message = TelegramFormatter.format_detection_message(
            species="European Hedgehog",
            confidence=0.87,
            motion_area=2500,
            timestamp="15:42"
        )
        
        assert "European Hedgehog" in message
        assert "87%" in message
        assert "2,500" in message
        assert "15:42" in message
        assert "🦔" in message  # Should include animal emoji
    
    def test_format_unknown_species_message(self):
        """Test unknown species message formatting."""
        message = TelegramFormatter.format_detection_message(
            species="Unknown species",
            confidence=0.0,
            motion_area=1800,
            timestamp="12:30"
        )
        
        assert "Unknown species" in message
        assert "1,800" in message
        assert "12:30" in message
        assert "🔍" in message  # Search emoji for unknown
    
    def test_format_system_message(self):
        """Test system message formatting."""
        message = TelegramFormatter.format_system_message(
            "Camera system started",
            level="info"
        )
        
        assert "Camera system started" in message
        assert "ℹ️" in message  # Info emoji
        
        # Test error message
        error_message = TelegramFormatter.format_system_message(
            "Camera initialization failed",
            level="error"
        )
        
        assert "❌" in error_message
    
    def test_format_statistics_message(self):
        """Test statistics message formatting."""
        stats = {
            'total_detections': 45,
            'unique_species': 3,
            'uptime_hours': 24,
            'memory_usage': 65.5
        }
        
        message = TelegramFormatter.format_statistics_message(stats)
        
        assert "45" in message
        assert "3" in message
        assert "24" in message
        assert "65.5%" in message
        assert "📊" in message  # Statistics emoji
    
    def test_escape_markdown(self):
        """Test markdown escaping."""
        test_text = "Test_with_underscores*and*asterisks[and]brackets"
        escaped = TelegramFormatter.escape_markdown(test_text)
        
        assert "\\_" in escaped
        assert "\\*" in escaped
        assert "\\[" in escaped
        assert "\\]" in escaped
    
    def test_truncate_message(self):
        """Test message truncation."""
        long_message = "A" * 5000  # Very long message
        truncated = TelegramFormatter.truncate_message(long_message, max_length=1000)
        
        assert len(truncated) <= 1000
        assert truncated.endswith("...")


class TestSystemMonitor:
    """Test system monitoring utilities."""
    
    def test_get_system_info(self):
        """Test system information gathering."""
        info = SystemMonitor.get_system_info()
        
        assert isinstance(info, dict)
        assert 'cpu_percent' in info
        assert 'memory' in info
        assert 'disk' in info
        assert 'temperature' in info
        assert 'uptime' in info
        
        assert 0 <= info['cpu_percent'] <= 100
        assert info['memory']['percent'] >= 0
    
    def test_check_system_health(self):
        """Test system health check."""
        health = SystemMonitor.check_system_health()
        
        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'issues' in health
        assert 'recommendations' in health
        
        assert health['overall_status'] in ['healthy', 'warning', 'critical']
        assert isinstance(health['issues'], list)
        assert isinstance(health['recommendations'], list)
    
    def test_get_process_info(self):
        """Test process information gathering."""
        info = SystemMonitor.get_process_info()
        
        assert isinstance(info, dict)
        assert 'pid' in info
        assert 'memory_mb' in info
        assert 'cpu_percent' in info
        assert 'threads' in info
        
        assert info['pid'] > 0
        assert info['memory_mb'] > 0
        assert info['threads'] >= 1
    
    @patch('psutil.sensors_temperatures')
    def test_get_temperature_info(self, mock_temps):
        """Test temperature information retrieval."""
        # Mock temperature data
        mock_temps.return_value = {
            'cpu_thermal': [Mock(label='', current=45.0, high=85.0, critical=95.0)]
        }
        
        temp_info = SystemMonitor.get_temperature_info()
        
        assert isinstance(temp_info, dict)
        if temp_info:  # Temperature might not be available on all systems
            assert 'cpu' in temp_info
            assert temp_info['cpu']['current'] == 45.0


class TestPerformanceTracker:
    """Test performance tracking utilities."""
    
    def test_performance_tracker_basic(self):
        """Test basic performance tracking."""
        tracker = PerformanceTracker()
        
        with tracker.track_operation("test_operation"):
            time.sleep(0.01)  # Small delay
        
        stats = tracker.get_stats()
        
        assert 'test_operation' in stats
        assert stats['test_operation']['count'] == 1
        assert stats['test_operation']['total_time'] > 0
        assert stats['test_operation']['average_time'] > 0
    
    def test_performance_tracker_multiple_operations(self):
        """Test tracking multiple operations."""
        tracker = PerformanceTracker()
        
        # Track multiple operations
        for i in range(3):
            with tracker.track_operation("operation_a"):
                time.sleep(0.01)
            
            with tracker.track_operation("operation_b"):
                time.sleep(0.005)
        
        stats = tracker.get_stats()
        
        assert stats['operation_a']['count'] == 3
        assert stats['operation_b']['count'] == 3
        assert stats['operation_a']['average_time'] > stats['operation_b']['average_time']
    
    def test_performance_tracker_reset(self):
        """Test performance tracker reset."""
        tracker = PerformanceTracker()
        
        with tracker.track_operation("test_op"):
            time.sleep(0.01)
        
        assert len(tracker.get_stats()) == 1
        
        tracker.reset()
        
        assert len(tracker.get_stats()) == 0
    
    def test_performance_tracker_summary(self):
        """Test performance summary generation."""
        tracker = PerformanceTracker()
        
        # Add some operations
        for _ in range(5):
            with tracker.track_operation("fast_op"):
                time.sleep(0.001)
            
            with tracker.track_operation("slow_op"):
                time.sleep(0.01)
        
        summary = tracker.get_summary()
        
        assert 'total_operations' in summary
        assert 'total_time' in summary
        assert 'slowest_operation' in summary
        assert 'fastest_operation' in summary
        
        assert summary['total_operations'] == 10
        assert summary['slowest_operation'] == 'slow_op'
        assert summary['fastest_operation'] == 'fast_op'


class TestUtilsErrorHandling:
    """Test error handling in utility functions."""
    
    def test_memory_manager_error_handling(self):
        """Test memory manager error handling."""
        # Test with mock that raises exception
        with patch('psutil.virtual_memory', side_effect=Exception("Memory error")):
            usage = MemoryManager.get_memory_usage()
            # Should return fallback values
            assert isinstance(usage, dict)
            assert 'error' in usage or usage['total_mb'] == 0
    
    def test_file_manager_error_handling(self):
        """Test file manager error handling."""
        # Test with invalid directory
        with pytest.raises(UtilsError):
            FileManager.get_directory_size(Path("/nonexistent/path"))
    
    def test_system_monitor_error_handling(self):
        """Test system monitor error handling."""
        # Test with mock that fails
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            info = SystemMonitor.get_system_info()
            # Should handle error gracefully
            assert isinstance(info, dict)
            # CPU info might be missing or have error indication


if __name__ == '__main__':
    pytest.main([__file__])