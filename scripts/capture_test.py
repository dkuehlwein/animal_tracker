#!/usr/bin/env python3
"""
Headless camera test and capture script for Pi Zero over SSH.

Features:
- Headless operation (no GUI)
- High-resolution image capture
- Real-time camera statistics
- Proper error handling and cleanup
- Direct picamera2 integration for better control

Commands:
- Press ENTER to capture image
- Type 'q' + ENTER to quit
- Type 'c' + ENTER for camera info
- Type 's' + ENTER for statistics
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Add system packages for libcamera access
sys.path.insert(0, '/usr/lib/python3/dist-packages')

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
    """Headless camera test application for Pi Zero over SSH."""
    
    def __init__(self):
        self.config = Config.create_test_config()
        self.camera_manager: Optional[CameraManager] = None
        self.capture_count = 0
        self.start_time = time.time()
        
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
        print(f"   Images per minute: {(self.capture_count / runtime * 60):.1f}")
    
    def run(self):
        """Main headless application loop."""
        print("🎥 Headless Camera Test & Capture Tool")
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
        
        try:
            print("\n🚀 Camera ready!")
            self._print_help()
            
            while True:
                try:
                    user_input = input("\nCommand (ENTER=capture, q=quit, c=info, s=stats, h=help): ").strip().lower()
                    
                    if user_input == 'q':
                        break
                    elif user_input == 'c':
                        self._print_camera_info()
                    elif user_input == 's':
                        self._print_statistics()
                    elif user_input == 'h':
                        self._print_help()
                    elif user_input == '' or user_input == 'capture':
                        self._capture_image()
                    else:
                        print("Unknown command. Type 'h' for help.")
                        
                except EOFError:
                    break
                    
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
        print("\n📖 HELP - Available Commands:")
        print("   ENTER (or 'capture') - Capture high-resolution image")
        print("   q                    - Exit application") 
        print("   c                    - Show camera information")
        print("   s                    - Show session statistics")
        print("   h                    - Show this help")
        print()
    
    def _cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        if self.camera_manager:
            try:
                self.camera_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
        
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