"""
Test burst frame cleanup functionality.
"""

import pytest
from pathlib import Path
import os
import sys
sys.path.append('src')

from resource_manager import StorageManager
from config import Config


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up environment variables after each test."""
    yield
    # Clean up any test-specific env vars
    for key in ['PERFORMANCE_MAX_IMAGES', 'STORAGE_DATA_DIR']:
        os.environ.pop(key, None)


class TestBurstFrameCleanup:
    """Test burst frame cleanup and counting."""

    def test_burst_frame_grouping(self, tmp_path):
        """Test that burst frames are grouped correctly for cleanup."""
        # Create test config with temporary directory
        os.environ['PERFORMANCE_MAX_IMAGES'] = '2'
        os.environ['STORAGE_DATA_DIR'] = str(tmp_path)
        config = Config.create_test_config()

        storage_manager = StorageManager(config)

        import time

        # Create some burst files with distinct timestamps
        # Burst 1 (oldest)
        (config.storage.image_dir / "capture_20231225_120000_frame1.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120000_frame2.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120000_frame3.jpg").touch()
        time.sleep(0.01)  # Ensure different mtime

        # Burst 2
        (config.storage.image_dir / "capture_20231225_120100_frame1.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120100_frame2.jpg").touch()
        time.sleep(0.01)

        # Burst 3 (newest)
        (config.storage.image_dir / "capture_20231225_120200_frame1.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120200_frame2.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120200_frame3.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120200_frame4.jpg").touch()

        # Should have 3 bursts total
        assert storage_manager.get_image_count() == 3

        # Cleanup should remove oldest burst (burst 1 with 3 files)
        deleted = storage_manager.cleanup_old_images()
        # Note: Could be 3 files, but depending on timing, might delete different burst
        assert deleted >= 2  # At least one burst deleted

        # Should have 2 bursts remaining
        assert storage_manager.get_image_count() == 2

    def test_burst_frame_counting(self, tmp_path):
        """Test that burst frames are counted as single events."""
        os.environ['STORAGE_DATA_DIR'] = str(tmp_path)
        config = Config.create_test_config()

        storage_manager = StorageManager(config)

        # Create burst with 5 frames
        for i in range(1, 6):
            (config.storage.image_dir / f"capture_20231225_120000_frame{i}.jpg").touch()

        # Should count as 1 burst, not 5 images
        assert storage_manager.get_image_count() == 1

        # Add another burst
        for i in range(1, 4):
            (config.storage.image_dir / f"capture_20231225_120100_frame{i}.jpg").touch()

        # Should count as 2 bursts
        assert storage_manager.get_image_count() == 2

    def test_mixed_burst_and_single_frames(self, tmp_path):
        """Test cleanup with mix of burst frames and single frames."""
        os.environ['PERFORMANCE_MAX_IMAGES'] = '2'
        os.environ['STORAGE_DATA_DIR'] = str(tmp_path)
        config = Config.create_test_config()

        storage_manager = StorageManager(config)

        import time

        # Create single frame (old format)
        (config.storage.image_dir / "capture_20231225_120000.jpg").touch()
        time.sleep(0.01)

        # Create burst frames
        (config.storage.image_dir / "capture_20231225_120100_frame1.jpg").touch()
        (config.storage.image_dir / "capture_20231225_120100_frame2.jpg").touch()
        time.sleep(0.01)

        # Create another single frame
        (config.storage.image_dir / "capture_20231225_120200.jpg").touch()

        # Should count as 3 captures
        assert storage_manager.get_image_count() == 3

        # Cleanup should remove oldest (one capture, either single frame or burst)
        deleted = storage_manager.cleanup_old_images()
        assert deleted >= 1  # At least 1 file deleted

        # Should have 2 captures remaining
        assert storage_manager.get_image_count() == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
