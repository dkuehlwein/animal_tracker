#!/usr/bin/env python3
"""
Direct Picamera2 test - bypass all our wrapper code.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add system packages for libcamera access
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
    print("Picamera2 imported successfully")
except ImportError as e:
    print(f"Cannot import Picamera2: {e}")
    sys.exit(1)

def test_direct_capture():
    """Test direct capture with Picamera2."""
    
    # Initialize camera
    camera = Picamera2()
    print("Camera initialized")
    
    # Create basic configuration
    config = camera.create_still_configuration(
        main={"size": (1920, 1080)},
        display=None
    )
    camera.configure(config)
    print("Camera configured")
    
    # Start camera
    camera.start()
    print("Camera started")
    
    try:
        # Wait for camera to settle
        import time
        time.sleep(2)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = Path("direct_test")
        test_dir.mkdir(exist_ok=True)
        file_path = test_dir / f"direct_{timestamp}.jpg"
        
        # Direct capture to file
        print(f"Capturing to {file_path}")
        camera.capture_file(str(file_path))
        
        print(f"✅ Direct capture saved: {file_path}")
        print(f"File size: {file_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error during capture: {e}")
    finally:
        camera.stop()
        camera.close()
        print("Camera stopped")

if __name__ == "__main__":
    test_direct_capture()