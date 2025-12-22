import gc
import logging
import psutil
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from astral import LocationInfo
from astral.sun import sun
from config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory management utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_threshold = config.performance.memory_threshold
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.5  # Default to 50% if unable to determine
    
    def is_memory_available(self):
        """Check if memory usage is below threshold"""
        return self.get_memory_usage() < self.memory_threshold
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        try:
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")
            return False
    
    def get_memory_info(self):
        """Get detailed memory information"""
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

class FileManager:
    """File management utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def cleanup_old_images(self):
        """Delete oldest images if exceeding max limit (includes annotated versions)"""
        try:
            # Get all original images (not annotated)
            original_images = sorted(
                [f for f in self.config.storage.image_dir.glob(f"{self.config.storage.image_prefix}*.jpg")
                 if "_annotated" not in f.stem],
                key=lambda x: x.stat().st_mtime
            )

            if len(original_images) > self.config.performance.max_images:
                files_to_delete = original_images[:(len(original_images) - self.config.performance.max_images)]
                deleted_count = 0

                for file_path in files_to_delete:
                    try:
                        # Delete the original image
                        file_path.unlink()
                        deleted_count += 1

                        # Also delete the annotated version if it exists
                        annotated_path = file_path.parent / f"{file_path.stem}_annotated{file_path.suffix}"
                        if annotated_path.exists():
                            annotated_path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}", exc_info=True)

                logger.info(f"Cleaned up {deleted_count} old images")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error in image cleanup: {e}", exc_info=True)
            return 0
    
    def get_storage_info(self):
        """Get storage space information"""
        try:
            usage = psutil.disk_usage(self.config.storage.data_dir)
            return {
                'total_mb': usage.total / (1024 * 1024),
                'used_mb': usage.used / (1024 * 1024),
                'free_mb': usage.free / (1024 * 1024),
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return None
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        try:
            self.config.storage.data_dir.mkdir(exist_ok=True)
            self.config.storage.image_dir.mkdir(exist_ok=True)
            self.config.storage.logs_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
    
    def get_image_count(self):
        """Get current number of stored images"""
        try:
            return len(list(self.config.storage.image_dir.glob(f"{self.config.storage.image_prefix}*.jpg")))
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            return 0

class SystemMonitor:
    """System monitoring utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.file_manager = FileManager(config)
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': self.memory_manager.get_memory_info(),
            'storage': self.file_manager.get_storage_info(),
            'image_count': self.file_manager.get_image_count(),
            'memory_available': self.memory_manager.is_memory_available()
        }
    
    def should_skip_processing(self):
        """Determine if processing should be skipped due to resource constraints"""
        if not self.memory_manager.is_memory_available():
            logger.warning(f"Skipping processing: Memory usage above {self.config.performance.memory_threshold*100}%")
            return True
        return False
    
    def log_system_status(self):
        """Log current system status"""
        status = self.get_system_status()
        if status['memory']:
            logger.info(f"Memory: {status['memory']['percent']:.1f}% used "
                        f"({status['memory']['available_mb']:.0f}MB available)")
        if status['storage']:
            logger.info(f"Storage: {status['storage']['percent']:.1f}% used "
                        f"({status['storage']['free_mb']:.0f}MB free)")
        logger.info(f"Images stored: {status['image_count']}")

class PerformanceTimer:
    """Performance timing utility"""
    
    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop timing and return duration"""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        return duration
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        duration = self.stop()
        if duration > 1.0:  # Log slow operations
            logger.info(f"{self.operation_name} took {duration:.2f}s")


class ImageUtils:
    """Image processing utilities."""

    @staticmethod
    def validate_image_path(image_path) -> bool:
        """Validate image file exists and has correct extension."""
        try:
            path = Path(image_path)
            return path.exists() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']
        except Exception:
            return False

    @staticmethod
    def get_image_info(image_path) -> dict:
        """Get image information."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {'error': 'Could not read image'}

            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1

            return {
                'width': width,
                'height': height,
                'channels': channels,
                'size_bytes': Path(image_path).stat().st_size
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def create_thumbnail(image_path, output_path, size=(128, 128)) -> bool:
        """Create thumbnail of image."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            thumbnail = cv2.resize(img, size)
            return cv2.imwrite(str(output_path), thumbnail)
        except Exception:
            return False


class MotionVisualizer:
    """Create annotated images showing motion detection regions."""

    @staticmethod
    def create_annotated_image(image_path: Path, motion_frame, config, motion_result) -> Path:
        """
        Create annotated version of image showing motion detection regions.

        Args:
            image_path: Path to the original high-res image
            motion_frame: The low-res grayscale frame used for motion detection
            config: System configuration
            motion_result: MotionResult object with detection details

        Returns:
            Path to the annotated image file
        """
        try:
            # Load the high-res image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None

            img_height, img_width = img.shape[:2]

            # Get motion detection parameters
            motion_width, motion_height = config.camera.motion_detection_resolution

            # Calculate scaling factors from motion frame to high-res image
            scale_x = img_width / motion_width
            scale_y = img_height / motion_height

            # Draw central region boundary (where motion is prioritized)
            bounds = config.motion.central_region_bounds
            center_x1 = int(img_width * bounds[0])
            center_x2 = int(img_width * bounds[1])
            center_y1 = int(img_height * bounds[0])
            center_y2 = int(img_height * bounds[1])

            # Draw semi-transparent overlay for central region
            overlay = img.copy()
            cv2.rectangle(overlay, (center_x1, center_y1), (center_x2, center_y2),
                         (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw the actual motion detection point if available
            if motion_result.center_x is not None and motion_result.center_y is not None:
                # Scale the detection point to high-res coordinates
                det_x = int(motion_result.center_x * scale_x)
                det_y = int(motion_result.center_y * scale_y)

                # Draw crosshair at detection point
                cv2.drawMarker(img, (det_x, det_y), (0, 0, 255),
                              markerType=cv2.MARKER_CROSS, markerSize=50, thickness=3)

                # Draw circle around detection point
                cv2.circle(img, (det_x, det_y), 30, (0, 0, 255), 3)

            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Motion info text
            text_lines = [
                f"Motion Area: {motion_result.motion_area} px",
                f"Threshold: {config.motion.motion_threshold} px",
                f"Contours: {motion_result.contour_count}",
                f"Confidence: {motion_result.detection_confidence:.1%}"
            ]

            # Add text with background for readability
            y_offset = 30
            for line in text_lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                # Draw black background rectangle
                cv2.rectangle(img, (5, y_offset - 25),
                            (text_size[0] + 15, y_offset + 5),
                            (0, 0, 0), -1)
                # Draw white text
                cv2.putText(img, line, (10, y_offset), font, font_scale,
                           (255, 255, 255), thickness)
                y_offset += 35

            # Add legend
            legend_y = img_height - 70
            cv2.rectangle(img, (5, legend_y - 25), (270, img_height - 10), (0, 0, 0), -1)
            cv2.putText(img, "Green: Central Region", (10, legend_y),
                       font, 0.5, (0, 255, 0), 1)
            cv2.putText(img, "Red: Motion Detection Point", (10, legend_y + 25),
                       font, 0.5, (0, 0, 255), 1)

            # Save annotated image with _annotated suffix
            annotated_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
            cv2.imwrite(str(annotated_path), img)

            logger.info(f"Created annotated image: {annotated_path}")
            return annotated_path

        except Exception as e:
            logger.error(f"Error creating annotated image: {e}", exc_info=True)
            return None


class TelegramFormatter:
    """Telegram message formatting utilities."""
    
    @staticmethod
    def format_detection_message(species_name: str, confidence: float, 
                               motion_area: int, timestamp: datetime) -> str:
        """Format detection message for Telegram."""
        time_str = timestamp.strftime("%H:%M")
        
        if species_name == "Unknown species":
            return f"🔍 Unknown species detected at {time_str}\nMotion area: {motion_area:,} pixels"
        
        # Emoji mapping
        emoji_map = {
            "Hedgehog": "🦔", "Fox": "🦊", "Squirrel": "🐿️", 
            "Cat": "🐱", "Bird": "🐦", "Robin": "🐦", "Blackbird": "🐦"
        }
        
        emoji = "🔍"
        for animal, symbol in emoji_map.items():
            if animal in species_name:
                emoji = symbol
                break
        
        if confidence > 0.8:
            return f"{emoji} {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}%"
        else:
            return f"🔍 Possible {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}%"
    
    @staticmethod
    def format_system_status(memory_percent: float, storage_percent: float,
                           image_count: int) -> str:
        """Format system status message."""
        return (f"📊 System Status\n"
                f"Memory: {memory_percent:.1f}% used\n"
                f"Storage: {storage_percent:.1f}% used\n"
                f"Images stored: {image_count}")


class PerformanceTracker:
    """Performance tracking and metrics utilities."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        """Start tracking an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str) -> float:
        """End tracking and return duration."""
        if operation_name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        
        self.metrics[operation_name].append(duration)
        del self.start_times[operation_name]
        
        return duration
    
    def get_average_time(self, operation_name: str) -> float:
        """Get average time for an operation."""
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return 0.0
        
        return sum(self.metrics[operation_name]) / len(self.metrics[operation_name])
    
    def get_total_operations(self, operation_name: str) -> int:
        """Get total number of operations."""
        return len(self.metrics.get(operation_name, []))
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


class SunChecker:
    """Check if it's currently daytime based on sunrise/sunset times."""
    
    def __init__(self, config: Config):
        self.config = config
        self.location = LocationInfo(
            name="Location",
            region=config.location.timezone,
            timezone=config.location.timezone,
            latitude=config.location.latitude,
            longitude=config.location.longitude
        )
        self._last_check_date = None
        self._sunrise = None
        self._sunset = None
        logger.info(f"SunChecker initialized for location: "
                   f"{config.location.latitude:.4f}, {config.location.longitude:.4f}")
    
    def _update_sun_times(self):
        """Update sunrise/sunset times for today."""
        from datetime import date, timezone
        try:
            today = date.today()
            if self._last_check_date != today:
                s = sun(self.location.observer, date=today)
                self._sunrise = s['sunrise']
                self._sunset = s['sunset']
                self._last_check_date = today
                logger.info(f"Sun times updated - Sunrise: {self._sunrise.strftime('%H:%M')}, "
                           f"Sunset: {self._sunset.strftime('%H:%M')}")
        except Exception as e:
            logger.error(f"Error calculating sun times: {e}")
            # Fallback to safe defaults (6am - 8pm local time)
            from datetime import datetime, time
            now = datetime.now()
            self._sunrise = datetime.combine(now.date(), time(6, 0))
            self._sunset = datetime.combine(now.date(), time(20, 0))
    
    def is_daytime(self) -> bool:
        """Check if it's currently daytime."""
        from datetime import datetime, timezone
        self._update_sun_times()
        
        now = datetime.now(timezone.utc)
        
        # Make sure sunrise and sunset are timezone-aware
        if self._sunrise and self._sunset:
            is_day = self._sunrise <= now <= self._sunset
            return is_day
        
        # Fallback if times aren't set
        return True
    
    def get_sun_info(self) -> dict:
        """Get sunrise/sunset information."""
        self._update_sun_times()
        return {
            'sunrise': self._sunrise.strftime('%H:%M') if self._sunrise else 'Unknown',
            'sunset': self._sunset.strftime('%H:%M') if self._sunset else 'Unknown',
            'is_daytime': self.is_daytime()
        }
        self.start_times.clear()