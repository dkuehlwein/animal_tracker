"""
Resource management for the wildlife detection system.

Consolidates memory management, file/storage management, and system monitoring
into a cohesive module for Raspberry Pi resource optimization.
"""

import gc
import logging
import psutil
from datetime import datetime, timedelta
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

    def __init__(self, config: Config, database=None):
        self.config = config
        # Optional DatabaseManager reference used only by purge_human_bursts()
        # (Task 3 privacy purge). None keeps legacy call sites (and tests)
        # working unchanged; the purge is simply a no-op without it.
        self.database = database

    def cleanup_old_images(self) -> int:
        """
        Delete oldest image bursts if exceeding max limit, and purge any
        HUMAN-status bursts past the privacy retention window.
        Groups burst frames together (e.g., capture_20231225_120000_frame1.jpg, _frame2.jpg, etc.)
        and treats each burst as a single unit for cleanup purposes.
        """
        self.purge_human_bursts()
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

    def purge_human_bursts(self) -> int:
        """
        Delete the saved photos of HUMAN-status detections older than
        `human_retention_hours` (privacy purge, Task 3), AND the saved
        photos of review-class (no_animal/unclassifiable) bursts that sit
        within `human_retention_proximity_seconds` of a HUMAN-status
        detection (leading-edge fix, 2026-07-31 — a burst just BEFORE the
        first HUMAN burst of a visit can leak a recognisable face to REVIEW
        and would otherwise survive the full ~300-burst rotation). The DB
        row for each detection is deliberately kept as a metadata-only
        record in both cases — only the image files are removed.

        A no-op if this StorageManager wasn't given a database reference, or
        if no burst files remain (idempotent when files are already gone).
        The proximity sweep is itself a no-op when
        human_retention_proximity_seconds <= 0 (rollback lever;
        DatabaseManager.get_human_adjacent_review_detections returns []
        without querying).
        """
        if self.database is None:
            return 0

        try:
            cutoff = datetime.now() - timedelta(hours=self.config.performance.human_retention_hours)

            human_rows = self.database.get_human_detections_older_than(cutoff)
            human_deleted = 0
            for _detection_id, image_path, _timestamp in human_rows:
                base = self._burst_base(image_path)
                human_deleted += self._delete_burst_files(base)

            proximity_rows = self.database.get_human_adjacent_review_detections(
                cutoff, self.config.performance.human_retention_proximity_seconds
            )
            proximity_deleted = 0
            for _detection_id, image_path, _timestamp in proximity_rows:
                base = self._burst_base(image_path)
                proximity_deleted += self._delete_burst_files(base)

            if human_rows:
                logger.info(
                    f"Purged {len(human_rows)} human-status burst(s) "
                    f"({human_deleted} files) older than "
                    f"{self.config.performance.human_retention_hours}h"
                )
            if proximity_rows:
                logger.info(
                    f"Purged {len(proximity_rows)} human-adjacent review burst(s) "
                    f"({proximity_deleted} files) within "
                    f"{self.config.performance.human_retention_proximity_seconds:.0f}s "
                    f"of a human detection, older than "
                    f"{self.config.performance.human_retention_hours}h"
                )
            return human_deleted + proximity_deleted

        except Exception as e:
            logger.error(f"Error purging human bursts: {e}", exc_info=True)
            return 0

    @staticmethod
    def _burst_base(image_path) -> str:
        """Derive a burst's base name from any one of its frame paths.

        e.g. ".../capture_20231225_120000_frame1.jpg" -> "capture_20231225_120000"
        """
        stem = Path(image_path).stem
        if "_frame" in stem:
            return stem.rsplit("_frame", 1)[0]
        return stem

    def _delete_burst_files(self, base: str) -> int:
        """Delete every file belonging to one burst (frames + annotated
        variants), constrained to image_dir. Idempotent: files that are
        already missing are simply skipped, never raised on.
        """
        deleted_count = 0
        image_dir = self.config.storage.image_dir

        # Burst frames (capture_TS_frameN.jpg) and old-format single files
        # (capture_TS.jpg) that happen to still be on disk for this base.
        candidates = [
            f for f in image_dir.glob(f"{base}*.jpg")
            if "_annotated" not in f.stem
        ]

        for file_path in candidates:
            try:
                file_path.unlink()
                deleted_count += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}", exc_info=True)
                continue

            annotated_path = file_path.parent / f"{file_path.stem}_annotated{file_path.suffix}"
            try:
                if annotated_path.exists():
                    annotated_path.unlink()
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {annotated_path}: {e}", exc_info=True)

        return deleted_count

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
