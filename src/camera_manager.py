import cv2
import time
from picamera2 import Picamera2
from config import Config
from datetime import datetime
from pathlib import Path

class CameraManager:
    def __init__(self, config: Config):
        self.config = config
        self.camera = None
        self.setup_camera()

    def setup_camera(self):
        """Initialize and configure the camera"""
        self.camera = Picamera2()
        
        # Configure dual streams
        preview_config = self.camera.create_preview_configuration(
            main={
                "size": self.config.main_resolution,
                "format": self.config.frame_format
            },
            lores={
                "size": self.config.lores_resolution,
                "format": self.config.frame_format
            }
        )
        self.camera.configure(preview_config)
        
        # Set frame rate limit
        self.camera.set_controls({
            "FrameDurationLimits": (
                self.config.frame_duration,
                self.config.frame_duration
            )
        })
        
        # Start camera
        self.camera.start()
        time.sleep(self.config.startup_delay)

    def capture_motion_frame(self):
        """Capture a low-resolution frame for motion detection"""
        return self.camera.capture_array("lores")

    def capture_photo(self):
        """Capture and save a high-resolution photo"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.config.data_dir / f"{self.config.image_prefix}{timestamp}.jpg"
            
            # Capture and save frame
            frame = self.camera.capture_array("main")
            cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            return image_path
        except Exception as e:
            print(f"Error capturing photo: {e}")
            return None

    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            self.camera.stop()
            self.camera = None 