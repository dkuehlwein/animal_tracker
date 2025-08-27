#!/usr/bin/env python3
"""
Camera test and focus adjustment script.

Features:
- Live preview with camera settings display
- High-resolution capture on spacebar
- Real-time camera statistics
- Proper error handling and cleanup
- Direct picamera2 integration for better control

Controls:
- SPACE: Capture high-res image
- ESC/Q: Exit
- C: Display camera info
- S: Show statistics
"""

import cv2
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from config import Config
    from camera_manager import CameraManager, CameraError
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Try to import picamera2 for direct camera detection
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class CameraTestApp:
    """Enhanced camera test application with better control and feedback."""
    
    def __init__(self):
        self.config = Config()
        self.camera_manager: Optional[CameraManager] = None
        self.capture_count = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Determine camera type
        self.use_mock = not self._is_pi_camera_available()
        
    def _is_pi_camera_available(self) -> bool:
        """Check if Pi camera is actually available."""
        if not PICAMERA2_AVAILABLE:
            logger.info("Picamera2 not available - using mock camera")
            return False
            
        try:
            # Quick test to see if camera hardware is accessible
            test_cam = Picamera2()
            camera_info = test_cam.global_camera_info()
            test_cam.close()
            
            if not camera_info:
                logger.info("No Pi camera detected - using mock camera")
                return False
                
            logger.info(f"Pi camera detected: {len(camera_info)} camera(s)")
            return True
            
        except Exception as e:
            logger.info(f"Pi camera not accessible ({e}) - using mock camera")
            return False
    
    def _update_fps(self):
        """Calculate and update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _draw_overlay(self, frame) -> None:
        """Draw informational overlay on frame."""
        if frame is None or len(frame.shape) < 2:
            return
            
        height, width = frame.shape[:2]
        
        # Prepare overlay text
        overlay_lines = [
            f"FPS: {self.fps:.1f}",
            f"Captures: {self.capture_count}",
            f"Resolution: {width}x{height}",
            f"Camera: {'Pi Camera' if not self.use_mock else 'Mock'}",
            "",
            "SPACE: Capture  ESC/Q: Exit",
            "C: Camera Info  S: Stats"
        ]
        
        # Draw semi-transparent background
        overlay_height = len(overlay_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (255, 255, 255), 1)
        
        # Draw text
        for i, line in enumerate(overlay_lines):
            y_pos = 30 + i * 25
            color = (0, 255, 0) if line.startswith(("FPS", "Captures")) else (255, 255, 255)
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def _print_camera_info(self):
        """Print detailed camera information."""
        print("\n" + "="*50)
        print("CAMERA INFORMATION")
        print("="*50)
        
        if self.camera_manager:
            info = self.camera_manager.get_system_info()
            print(f"Camera Type: {info['camera_type']}")
            print(f"Main Resolution: {info['configuration']['main_resolution']}")
            print(f"Motion Resolution: {info['configuration']['motion_resolution']}")
            print(f"Image Directory: {info['storage']['image_dir']}")
            print(f"Image Prefix: {info['storage']['image_prefix']}")
            
            stats = info['stats']
            print(f"\nCamera Status:")
            print(f"  Running: {stats['is_running']}")
            print(f"  Error Count: {stats['error_count']}")
            print(f"  Active Frames: {stats['active_frames']}")
        
        print("="*50)
    
    def _print_statistics(self):
        """Print session statistics."""
        runtime = time.time() - self.start_time
        print(f"\n📊 Session Statistics:")
        print(f"   Runtime: {runtime:.1f} seconds")
        print(f"   Images Captured: {self.capture_count}")
        print(f"   Average FPS: {self.fps:.1f}")
        print(f"   Images per minute: {(self.capture_count / runtime * 60):.1f}")
    
    def run(self):
        """Main application loop."""
        print("🎥 Camera Test & Focus Adjustment Tool")
        print("="*50)
        
        if self.use_mock:
            print("⚠️  Using MOCK camera (Pi camera not available)")
        else:
            print("📷 Using Pi Camera")
            
        print(f"💾 Images will be saved to: {self.config.storage.image_dir}")
        print("="*50)
        
        # Initialize camera
        try:
            self.camera_manager = CameraManager(self.config, use_mock=self.use_mock)
            self.camera_manager.start()
            logger.info("Camera started successfully")
        except CameraError as e:
            logger.error(f"Failed to initialize camera: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error during camera initialization: {e}")
            return 1
        
        # Setup display
        window_name = 'Camera Test - Press SPACE to capture'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        try:
            print("\n🚀 Camera ready! Press 'h' for help")
            
            while True:
                # Capture preview frame
                frame = self.camera_manager.capture_motion_frame()
                
                if frame is not None:
                    # Convert grayscale to BGR for display
                    if len(frame.shape) == 2:
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        display_frame = frame.copy()
                    
                    # Add overlay
                    self._draw_overlay(display_frame)
                    self._update_fps()
                    
                    # Display frame
                    cv2.imshow(window_name, display_frame)
                else:
                    logger.warning("Failed to capture frame")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar - capture
                    self._capture_image()
                    
                elif key in [27, ord('q'), ord('Q')]:  # ESC or Q - quit
                    break
                    
                elif key in [ord('c'), ord('C')]:  # Camera info
                    self._print_camera_info()
                    
                elif key in [ord('s'), ord('S')]:  # Statistics
                    self._print_statistics()
                    
                elif key in [ord('h'), ord('H')]:  # Help
                    self._print_help()
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            return 1
        finally:
            self._cleanup()
            
        return 0
    
    def _capture_image(self):
        """Capture and save high-resolution image."""
        print(f"\n📸 Capturing image {self.capture_count + 1}...")
        
        try:
            saved_path = self.camera_manager.capture_and_save_photo()
            
            if saved_path:
                self.capture_count += 1
                file_size = saved_path.stat().st_size / 1024  # KB
                print(f"✅ Image saved: {saved_path.name} ({file_size:.1f} KB)")
            else:
                print("❌ Failed to capture image")
                
        except Exception as e:
            logger.error(f"Error during image capture: {e}")
            print("❌ Capture failed due to error")
    
    def _print_help(self):
        """Print help information."""
        print("\n📖 HELP - Keyboard Controls:")
        print("   SPACE    - Capture high-resolution image")
        print("   ESC/Q    - Exit application") 
        print("   C        - Show camera information")
        print("   S        - Show session statistics")
        print("   H        - Show this help")
        print()
    
    def _cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        if self.camera_manager:
            try:
                self.camera_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
                
        cv2.destroyAllWindows()
        
        # Final statistics
        runtime = time.time() - self.start_time
        print(f"\n📊 Final Statistics:")
        print(f"   Session duration: {runtime:.1f} seconds")
        print(f"   Total captures: {self.capture_count}")
        print("   Thank you for using Camera Test Tool! 👋")


def main() -> int:
    """Main entry point."""
    app = CameraTestApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())