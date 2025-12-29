"""
Camera management for Wildlife Detector system.
Provides robust camera control with error handling, resource management, and monitoring.
"""

from __future__ import annotations
import cv2
import time
import gc
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Protocol
from contextlib import contextmanager
import logging

from config import Config
from exceptions import CameraError, CameraInitializationError, CameraOperationError

logger = logging.getLogger(__name__)


class FrameData(Protocol):
    """Protocol for frame data."""
    @property
    def shape(self) -> Tuple[int, ...]:
        """Frame shape."""
        ...

    def copy(self) -> 'FrameData':
        """Create a copy of the frame."""
        ...


class CameraInterface(ABC):
    """Abstract interface for camera implementations."""

    @abstractmethod
    def start(self) -> None:
        """Start the camera."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera."""
        pass

    @abstractmethod
    def capture_motion_frame(self) -> Optional[FrameData]:
        """Capture frame for motion detection."""
        pass

    @abstractmethod
    def capture_high_res_frame(self) -> Optional[FrameData]:
        """Capture high resolution frame."""
        pass

    @abstractmethod
    def capture_burst_frames(self, count: int, interval: float) -> list:
        """Capture multiple high-resolution frames in quick succession."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if camera is available."""
        pass


class ResourceManager:
    """Manages camera resources and memory cleanup."""
    
    def __init__(self):
        self._active_frames = set()
        self._lock = threading.Lock()
    
    def register_frame(self, frame_id: str):
        """Register an active frame."""
        with self._lock:
            self._active_frames.add(frame_id)
    
    def unregister_frame(self, frame_id: str):
        """Unregister an active frame."""
        with self._lock:
            self._active_frames.discard(frame_id)
    
    def cleanup_all(self):
        """Force cleanup of all registered frames."""
        with self._lock:
            self._active_frames.clear()
            gc.collect()
    
    def get_active_count(self) -> int:
        """Get number of active frames."""
        with self._lock:
            return len(self._active_frames)


class PiCameraManager(CameraInterface):
    """
    Production camera manager using Picamera2.
    Optimized for Pi Zero 2 W with robust error handling.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = None
        self._is_running = False
        self._resource_manager = ResourceManager()
        self._last_error_time = 0
        self._error_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0
        
        logger.info("Initializing PiCamera manager")
    
    def start(self) -> None:
        """Initialize and start the camera with retry logic."""
        if self._is_running:
            logger.warning("Camera is already running")
            return
        
        retry_count = 0
        while retry_count < self._max_retries:
            try:
                self._initialize_camera()
                self._is_running = True
                logger.info("Camera started successfully")
                return
            except Exception as e:
                retry_count += 1
                logger.error(f"Camera initialization attempt {retry_count} failed: {e}")
                if retry_count < self._max_retries:
                    time.sleep(self._retry_delay * retry_count)
                else:
                    raise CameraInitializationError(f"Failed to initialize camera after {self._max_retries} attempts") from e
    
    def _initialize_camera(self) -> None:
        """Initialize camera with configuration."""
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise CameraInitializationError("Picamera2 library not available")
        
        self.camera = Picamera2()
        
        # Configure dual streams
        try:
            preview_config = self.camera.create_preview_configuration(
                main={
                    "size": self.config.camera.main_resolution,
                    "format": self.config.camera.frame_format
                },
                lores={
                    "size": self.config.camera.motion_detection_resolution,
                    "format": self.config.camera.motion_detection_format
                }
            )
            self.camera.configure(preview_config)
        except Exception as e:
            raise CameraInitializationError(f"Failed to configure camera streams: {e}") from e

        # Set camera controls BEFORE starting camera
        try:
            controls = {
                "FrameDurationLimits": (
                    self.config.camera.frame_duration,
                    self.config.camera.frame_duration
                )
            }

            # Manual exposure mode: disable AE/AGC and set fixed values
            if self.config.camera.exposure_time is not None and self.config.camera.analogue_gain is not None:
                controls["AeEnable"] = False  # Disable auto-exposure
                controls["ExposureTime"] = self.config.camera.exposure_time
                controls["AnalogueGain"] = self.config.camera.analogue_gain
                logger.info(f"Manual exposure mode: ExposureTime={self.config.camera.exposure_time}μs "
                           f"(1/{1000000/self.config.camera.exposure_time:.0f}s), "
                           f"Gain={self.config.camera.analogue_gain}x")

            self.camera.set_controls(controls)
        except Exception as e:
            logger.warning(f"Failed to set camera controls: {e}")

        # Start camera
        try:
            self.camera.start()
            time.sleep(self.config.camera.startup_delay)
        except Exception as e:
            raise CameraInitializationError(f"Failed to start camera: {e}") from e
    
    def stop(self) -> None:
        """Stop the camera and cleanup resources."""
        if not self._is_running:
            return
        
        try:
            if self.camera:
                self.camera.stop()
                self.camera = None
            self._resource_manager.cleanup_all()
            self._is_running = False
            logger.info("Camera stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def capture_motion_frame(self) -> Optional[FrameData]:
        """Capture frame for motion detection (RGB or grayscale based on config)."""
        if not self._is_running or not self.camera:
            raise CameraOperationError("Camera not initialized")

        try:
            frame_id = f"motion_{time.time()}"
            self._resource_manager.register_frame(frame_id)

            # Use RGB for motion detection if color filtering is enabled
            if self.config.motion.enable_color_filtering:
                # Capture RGB frame from lores stream
                # Picamera2 lores stream is configured as YUV420, need to convert
                yuv_frame = self.camera.capture_array("lores")
                h, w = self.config.camera.motion_detection_resolution[1], self.config.camera.motion_detection_resolution[0]

                # Convert YUV420 to BGR (OpenCV format)
                # YUV420 format: Y plane (h x w), followed by U and V planes (h/2 x w/2 each)
                yuv_full = yuv_frame.reshape((h * 3 // 2, w))
                bgr_frame = cv2.cvtColor(yuv_full, cv2.COLOR_YUV2BGR_I420)

                self._resource_manager.unregister_frame(frame_id)
                self._reset_error_count()

                return bgr_frame
            else:
                # Original grayscale (Y channel only) for motion detection
                yuv_frame = self.camera.capture_array("lores")

                # Extract Y channel (luminance) for motion detection
                h, w = self.config.camera.motion_detection_resolution[1], self.config.camera.motion_detection_resolution[0]
                y_channel = yuv_frame[:h, :w].copy()

                # Ensure proper data type
                if y_channel.dtype != 'uint8':
                    y_channel = y_channel.astype('uint8')

                self._resource_manager.unregister_frame(frame_id)
                self._reset_error_count()

                return y_channel

        except Exception as e:
            self._handle_capture_error(f"motion frame capture: {e}")
            return None
    
    def capture_high_res_frame(self) -> Optional[FrameData]:
        """Capture high resolution frame with memory management."""
        if not self._is_running or not self.camera:
            raise CameraOperationError("Camera not initialized")
        
        try:
            frame_id = f"highres_{time.time()}"
            self._resource_manager.register_frame(frame_id)
            
            # Capture high resolution frame
            frame = self.camera.capture_array("main")
            frame_copy = frame.copy()

            self._resource_manager.unregister_frame(frame_id)
            self._reset_error_count()
            
            return frame_copy
            
        except Exception as e:
            self._handle_capture_error(f"high-res frame capture: {e}")
            return None
    
    def capture_burst_frames(self, count: int, interval: float) -> list:
        """Capture multiple high-resolution frames in quick succession."""
        if not self._is_running or not self.camera:
            raise CameraOperationError("Camera not initialized")

        frames = []
        frame_metadata = []  # Diagnostic: track frame timing
        burst_start_time = time.time()

        try:
            logger.info(f"Starting burst capture: {count} frames at {interval}s intervals")

            # DIAGNOSTIC: Flush buffer with dummy capture
            flush_start = time.time()
            try:
                _ = self.camera.capture_array("main")
                flush_time_ms = (time.time() - flush_start) * 1000
                logger.info(f"[DIAG] Buffer flush: {flush_time_ms:.1f}ms")
            except Exception as e:
                logger.warning(f"[DIAG] Buffer flush failed: {e}")

            for i in range(count):
                frame_start = time.time()
                frame = self.capture_high_res_frame()
                capture_time_ms = (time.time() - frame_start) * 1000

                if frame is not None:
                    frames.append(frame)

                    # DIAGNOSTIC: Try to get frame metadata from camera
                    try:
                        metadata = self.camera.capture_metadata()
                        sensor_timestamp = metadata.get('SensorTimestamp', 0)
                        exposure_time = metadata.get('ExposureTime', 0)
                        frame_duration = metadata.get('FrameDuration', 0)

                        frame_info = {
                            'frame_num': i + 1,
                            'capture_time_ms': capture_time_ms,
                            'sensor_timestamp_us': sensor_timestamp / 1000 if sensor_timestamp else 0,
                            'exposure_us': exposure_time,
                            'frame_duration_us': frame_duration,
                            'time_since_burst_start_ms': (time.time() - burst_start_time) * 1000
                        }
                        frame_metadata.append(frame_info)

                        logger.info(f"[DIAG] Frame {i+1}/{count}: capture={capture_time_ms:.1f}ms, "
                                   f"sensor_ts={sensor_timestamp/1e6:.3f}s, "
                                   f"exposure={exposure_time}us, "
                                   f"since_burst={frame_info['time_since_burst_start_ms']:.1f}ms")
                    except Exception as e:
                        logger.debug(f"[DIAG] Could not get metadata for frame {i+1}: {e}")
                        frame_metadata.append({'frame_num': i + 1, 'capture_time_ms': capture_time_ms})
                else:
                    logger.warning(f"Failed to capture burst frame {i+1}/{count}")

                # Wait between frames (except after last frame)
                if i < count - 1 and interval > 0:
                    time.sleep(interval)

            total_burst_time = (time.time() - burst_start_time) * 1000
            logger.info(f"Burst capture complete: {len(frames)}/{count} frames in {total_burst_time:.1f}ms")

            # DIAGNOSTIC: Analyze frame timing for staleness
            if len(frame_metadata) >= 2:
                sensor_times = [m.get('sensor_timestamp_us', 0) for m in frame_metadata if m.get('sensor_timestamp_us')]
                if len(sensor_times) >= 2:
                    time_gaps = [sensor_times[i+1] - sensor_times[i] for i in range(len(sensor_times)-1)]
                    avg_gap_ms = sum(time_gaps) / len(time_gaps) / 1000
                    logger.info(f"[DIAG] Sensor timestamp gaps (ms): {[f'{g/1000:.1f}' for g in time_gaps]}, avg={avg_gap_ms:.1f}ms")

                    # Check if first frame might be stale (sensor timestamp older than expected)
                    if sensor_times[0] > 0 and avg_gap_ms > 0:
                        expected_first_frame_age = avg_gap_ms  # Should be roughly one frame interval
                        if time_gaps[0] / 1000 > avg_gap_ms * 2:
                            logger.warning(f"[DIAG] First frame gap ({time_gaps[0]/1000:.1f}ms) is >2x average - possible stale buffer!")

            return frames

        except Exception as e:
            logger.error(f"Error during burst capture: {e}")
            return frames  # Return what we captured so far
    
    def save_frame_to_file(self, frame: FrameData, file_path: Path) -> bool:
        """Save frame to file with error handling."""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Picamera2 appears to already provide BGR format on this hardware
            # Skip color conversion to fix yellow->blue, red->purple color swap
            bgr_frame = frame

            # Save image (use default JPEG compression)
            success = cv2.imwrite(str(file_path), bgr_frame)

            if success:
                logger.debug(f"Frame saved to {file_path}")
            else:
                logger.error(f"Failed to save frame to {file_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving frame to {file_path}: {e}")
            return False

    def save_burst_frames(self, frames: list, base_path: Path) -> list:
        """
        Save all frames from a burst capture.

        Args:
            frames: List of frames to save
            base_path: Base path for saving (e.g., capture_20231225_120000.jpg)
                      Frames will be saved as: capture_20231225_120000_frame1.jpg, etc.

        Returns:
            List of paths to saved frames
        """
        saved_paths = []

        try:
            # Extract components from base_path
            stem = base_path.stem  # e.g., "capture_20231225_120000"
            suffix = base_path.suffix  # e.g., ".jpg"
            parent = base_path.parent

            for i, frame in enumerate(frames):
                # Create filename: capture_20231225_120000_frame1.jpg
                frame_path = parent / f"{stem}_frame{i+1}{suffix}"

                if self.save_frame_to_file(frame, frame_path):
                    saved_paths.append(frame_path)
                else:
                    logger.warning(f"Failed to save burst frame {i+1}/{len(frames)}")

            logger.info(f"Saved {len(saved_paths)}/{len(frames)} burst frames")
            return saved_paths

        except Exception as e:
            logger.error(f"Error saving burst frames: {e}")
            return saved_paths
    
    def is_available(self) -> bool:
        """Check if camera is available and operational."""
        try:
            return self._is_running and self.camera is not None
        except Exception:
            return False
    
    def get_stats(self) -> dict:
        """Get camera statistics."""
        return {
            "is_running": self._is_running,
            "error_count": self._error_count,
            "active_frames": self._resource_manager.get_active_count(),
            "last_error_time": self._last_error_time
        }
    
    def _handle_capture_error(self, error_msg: str) -> None:
        """Handle capture errors with automatic recovery."""
        self._error_count += 1
        self._last_error_time = time.time()
        logger.error(f"Camera capture error: {error_msg}")
        
        # Force cleanup on repeated errors
        if self._error_count > 5:
            logger.warning("Multiple camera errors, forcing resource cleanup")
            self._resource_manager.cleanup_all()
        
        # Attempt recovery if too many errors
        if self._error_count > 10:
            logger.warning("Too many camera errors, attempting restart")
            try:
                self.stop()
                time.sleep(2)
                self.start()
            except Exception as e:
                logger.error(f"Camera restart failed: {e}")
    
    def _reset_error_count(self) -> None:
        """Reset error count after successful operation."""
        if self._error_count > 0:
            self._error_count = max(0, self._error_count - 1)


class MockCameraManager(CameraInterface):
    """Mock camera for testing and development."""
    
    def __init__(self, config: Config):
        self.config = config
        self._is_running = False
        logger.info("Initializing Mock camera manager")
    
    def start(self) -> None:
        """Start mock camera."""
        self._is_running = True
        logger.info("Mock camera started")
    
    def stop(self) -> None:
        """Stop mock camera."""
        self._is_running = False
        logger.info("Mock camera stopped")
    
    def capture_motion_frame(self) -> Optional[FrameData]:
        """Return mock motion detection frame (RGB or grayscale based on config)."""
        if not self._is_running:
            return None

        import numpy as np
        h, w = self.config.camera.motion_detection_resolution[1], self.config.camera.motion_detection_resolution[0]

        if self.config.motion.enable_color_filtering:
            # Create mock RGB frame with some "motion"
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            return frame
        else:
            # Create mock grayscale frame with some "motion"
            frame = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            return frame
    
    def capture_high_res_frame(self) -> Optional[FrameData]:
        """Return mock high resolution frame."""
        if not self._is_running:
            return None
        
        import numpy as np
        h, w = self.config.camera.main_resolution[1], self.config.camera.main_resolution[0]
        
        # Create mock RGB frame
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        return frame
    
    def capture_burst_frames(self, count: int, interval: float) -> list:
        """Return mock burst frames with varying sharpness."""
        if not self._is_running:
            return []
        
        import numpy as np
        h, w = self.config.camera.main_resolution[1], self.config.camera.main_resolution[0]
        
        frames = []
        for i in range(count):
            # Create frames with varying sharpness (simulate real scenario)
            # Middle frames tend to be sharper
            sharpness_factor = 1.0 - abs(i - count // 2) * 0.15
            frame = np.random.randint(0, int(255 * sharpness_factor), (h, w, 3), dtype=np.uint8)
            frames.append(frame)
            
            if i < count - 1 and interval > 0:
                time.sleep(interval)
        
        return frames
    
    def save_frame_to_file(self, frame: FrameData, file_path: Path) -> bool:
        """Mock save frame to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()  # Create empty file
        return True
    
    def is_available(self) -> bool:
        """Mock camera is always available when running."""
        return self._is_running
    
    def get_stats(self) -> dict:
        """Get mock camera statistics."""
        return {
            "is_running": self._is_running,
            "error_count": 0,
            "active_frames": 0,
            "last_error_time": 0
        }


class CameraManager:
    """
    High-level camera manager that handles image capture and file operations.
    Provides a clean interface for the wildlife detection system.
    """
    
    def __init__(self, config: Config, use_mock: bool = False):
        self.config = config
        self._camera: CameraInterface = (
            MockCameraManager(config) if use_mock 
            else PiCameraManager(config)
        )
        
        logger.info(f"Camera manager initialized with {type(self._camera).__name__}")
    
    @contextmanager
    def camera_session(self):
        """Context manager for camera operations."""
        try:
            self._camera.start()
            yield self._camera
        finally:
            self._camera.stop()
    
    def capture_motion_frame(self) -> Optional[FrameData]:
        """Capture frame optimized for motion detection."""
        return self._camera.capture_motion_frame()
    
    def capture_and_save_photo(self) -> Optional[Path]:
        """Capture high-resolution photo and save to disk."""
        try:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config.storage.image_dir / f"{self.config.storage.image_prefix}{timestamp}.jpg"
            
            # Capture frame
            frame = self._camera.capture_high_res_frame()
            if frame is None:
                logger.error("Failed to capture high resolution frame")
                return None
            
            # Save to file using camera manager's optimized saving
            if isinstance(self._camera, PiCameraManager):
                success = self._camera.save_frame_to_file(frame, file_path)
            else:
                # For mock camera, use simple save
                success = self._camera.save_frame_to_file(frame, file_path)
            
            if success:
                logger.info(f"Photo saved: {file_path}")
                return file_path
            else:
                logger.error("Failed to save photo")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing and saving photo: {e}")
            return None
        finally:
            # Force cleanup for Pi Zero
            gc.collect()
    
    def capture_high_res_frame(self) -> Optional[FrameData]:
        """Capture high resolution frame."""
        return self._camera.capture_high_res_frame()

    def capture_burst_frames(self, count: int, interval: float) -> list:
        """Capture multiple high-resolution frames in quick succession."""
        return self._camera.capture_burst_frames(count, interval)

    def save_burst_frames(self, frames: list, base_path: Path) -> list:
        """
        Save all frames from a burst capture.
        Only works with PiCameraManager - returns empty list for mock camera.
        """
        if isinstance(self._camera, PiCameraManager):
            return self._camera.save_burst_frames(frames, base_path)
        else:
            # Mock camera - just return empty list
            logger.debug("Mock camera: skipping burst frame save")
            return []

    def start(self) -> None:
        """Start the camera system."""
        self._camera.start()

    def stop(self) -> None:
        """Stop the camera system."""
        self._camera.stop()

    def is_operational(self) -> bool:
        """Check if camera system is operational."""
        return self._camera.is_available()
    
    def get_system_info(self) -> dict:
        """Get comprehensive camera system information."""
        stats = self._camera.get_stats()
        return {
            "camera_type": type(self._camera).__name__,
            "configuration": {
                "main_resolution": self.config.camera.main_resolution,
                "motion_resolution": self.config.camera.motion_detection_resolution,
                "motion_format": self.config.camera.motion_detection_format
            },
            "stats": stats,
            "storage": {
                "image_dir": str(self.config.storage.image_dir),
                "image_prefix": self.config.storage.image_prefix
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()