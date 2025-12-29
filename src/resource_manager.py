"""
Resource management for the wildlife detection system.

Consolidates memory management, file/storage management, and system monitoring
into a cohesive module for Raspberry Pi resource optimization.
"""

import gc
import logging
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory management utilities for Raspberry Pi."""

    def __init__(self, config: Config):
        self.config = config
        self.memory_threshold = config.performance.memory_threshold

    def get_memory_usage(self) -> float:
        """Get current memory usage as a ratio (0.0 to 1.0)."""
        try:
            return psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.5  # Default to 50% if unable to determine

    def is_memory_available(self) -> bool:
        """Check if memory usage is below threshold."""
        return self.get_memory_usage() < self.memory_threshold

    def force_cleanup(self) -> bool:
        """Force garbage collection and cleanup."""
        try:
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")
            return False

    def get_memory_info(self) -> Optional[dict]:
        """Get detailed memory information."""
        try:
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total / (1024 * 1024),
                'available_mb': mem.available / (1024 * 1024),
                'used_mb': mem.used / (1024 * 1024),
                'percent': mem.percent,
                'free_mb': mem.free / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return None


class StorageManager:
    """File and storage management utilities."""

    def __init__(self, config: Config):
        self.config = config

    def cleanup_old_images(self) -> int:
        """
        Delete oldest image bursts if exceeding max limit.
        Groups burst frames together (e.g., capture_20231225_120000_frame1.jpg, _frame2.jpg, etc.)
        and treats each burst as a single unit for cleanup purposes.
        """
        try:
            # Get all capture files (excluding annotated versions)
            all_files = [
                f for f in self.config.storage.image_dir.glob(f"{self.config.storage.image_prefix}*.jpg")
                if "_annotated" not in f.stem
            ]

            # Group files by burst timestamp
            # Format: capture_20231225_120000_frame1.jpg -> base: capture_20231225_120000
            burst_groups = {}
            for file_path in all_files:
                # Extract base timestamp (remove _frameN suffix if present)
                stem = file_path.stem
                if "_frame" in stem:
                    # Burst frame: capture_20231225_120000_frame1 -> capture_20231225_120000
                    base = stem.rsplit("_frame", 1)[0]
                else:
                    # Single frame or old format: use stem as-is
                    base = stem

                if base not in burst_groups:
                    burst_groups[base] = []
                burst_groups[base].append(file_path)

            # Sort burst groups by oldest file in each group
            sorted_bursts = sorted(
                burst_groups.items(),
                key=lambda x: min(f.stat().st_mtime for f in x[1])
            )

            # Count how many bursts we have
            burst_count = len(sorted_bursts)

            if burst_count > self.config.performance.max_images:
                # Delete oldest bursts (including all their frames)
                bursts_to_delete = sorted_bursts[:(burst_count - self.config.performance.max_images)]
                deleted_count = 0

                for base, file_paths in bursts_to_delete:
                    for file_path in file_paths:
                        try:
                            # Delete the capture frame
                            file_path.unlink()
                            deleted_count += 1

                            # Also delete the annotated version if it exists
                            annotated_path = file_path.parent / f"{file_path.stem}_annotated{file_path.suffix}"
                            if annotated_path.exists():
                                annotated_path.unlink()
                                deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}", exc_info=True)

                logger.info(f"Cleaned up {len(bursts_to_delete)} old bursts ({deleted_count} files total)")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error in image cleanup: {e}", exc_info=True)
            return 0

    def get_storage_info(self) -> Optional[dict]:
        """Get storage space information."""
        try:
            usage = psutil.disk_usage(str(self.config.storage.data_dir))
            return {
                'total_mb': usage.total / (1024 * 1024),
                'used_mb': usage.used / (1024 * 1024),
                'free_mb': usage.free / (1024 * 1024),
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return None

    def ensure_directories(self) -> bool:
        """Ensure all required directories exist."""
        try:
            self.config.storage.data_dir.mkdir(exist_ok=True)
            self.config.storage.image_dir.mkdir(exist_ok=True)
            self.config.storage.logs_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False

    def get_image_count(self) -> int:
        """
        Get current number of stored image bursts (not individual frames).
        Counts each burst as a single detection event.
        """
        try:
            # Get all capture files (excluding annotated versions)
            all_files = [
                f for f in self.config.storage.image_dir.glob(f"{self.config.storage.image_prefix}*.jpg")
                if "_annotated" not in f.stem
            ]

            # Group by burst timestamp to count bursts, not individual frames
            burst_bases = set()
            for file_path in all_files:
                stem = file_path.stem
                if "_frame" in stem:
                    # Burst frame: extract base timestamp
                    base = stem.rsplit("_frame", 1)[0]
                else:
                    # Single frame: use stem as-is
                    base = stem
                burst_bases.add(base)

            return len(burst_bases)
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            return 0


class SystemMonitor:
    """
    Unified system resource monitoring for Raspberry Pi.

    Combines memory monitoring, storage management, and CPU temperature
    tracking into a single interface.
    """

    def __init__(self, config: Config):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.storage_manager = StorageManager(config)

    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': self.memory_manager.get_memory_info(),
            'storage': self.storage_manager.get_storage_info(),
            'image_count': self.storage_manager.get_image_count(),
            'memory_available': self.memory_manager.is_memory_available(),
            'cpu_temp': self.get_cpu_temperature()
        }

    def should_skip_processing(self) -> bool:
        """Determine if processing should be skipped due to resource constraints."""
        if not self.memory_manager.is_memory_available():
            logger.warning(f"Skipping processing: Memory usage above {self.config.performance.memory_threshold*100}%")
            return True
        return False

    def log_system_status(self) -> None:
        """Log current system status."""
        status = self.get_system_status()
        if status['memory']:
            logger.info(f"Memory: {status['memory']['percent']:.1f}% used "
                        f"({status['memory']['available_mb']:.0f}MB available)")
        if status['storage']:
            logger.info(f"Storage: {status['storage']['percent']:.1f}% used "
                        f"({status['storage']['free_mb']:.0f}MB free)")
        if status['cpu_temp']:
            logger.info(f"CPU Temp: {status['cpu_temp']:.1f}°C")
        logger.info(f"Images stored: {status['image_count']}")

    def get_cpu_temperature(self) -> Optional[float]:
        """Get Raspberry Pi CPU temperature. Returns None on error."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_str = f.read()
            return float(temp_str) / 1000.0
        except Exception as e:
            logger.error(f"Error reading CPU temperature: {e}")
            return None
