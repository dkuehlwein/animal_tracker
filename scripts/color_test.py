#!/usr/bin/env python3
"""
Color format diagnostic script to identify the correct color conversion.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from config import Config
    from camera_manager import CameraManager
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

def test_color_formats():
    """Test different color format conversions."""
    config = Config.create_test_config()
    
    # Determine if Pi camera is available
    try:
        from picamera2 import Picamera2
        use_mock = False
        print("Using Pi camera")
    except ImportError:
        use_mock = True
        print("Using mock camera for testing")
    
    camera_manager = CameraManager(config, use_mock=use_mock)
    
    try:
        camera_manager.start()
        print("Camera started successfully")
        
        # Capture a frame
        frame = camera_manager._camera.capture_high_res_frame()
        if frame is None:
            print("Failed to capture frame")
            return
        
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame min/max values: {frame.min()}/{frame.max()}")
        print(f"Frame mean: {frame.mean():.2f}")
        
        # Check if frame data looks valid
        if len(frame.shape) == 3:
            print(f"Channel shapes: R={frame[:,:,0].shape}, G={frame[:,:,1].shape}, B={frame[:,:,2].shape}")
            print(f"Channel means: R={frame[:,:,0].mean():.2f}, G={frame[:,:,1].mean():.2f}, B={frame[:,:,2].mean():.2f}")
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Test different conversion approaches
            test_dir = Path("color_test")
            test_dir.mkdir(exist_ok=True)
            
            # Save without conversion
            cv2.imwrite(str(test_dir / "no_conversion.jpg"), frame)
            print("Saved: no_conversion.jpg (raw frame)")
            
            # Save with RGB->BGR conversion 
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(test_dir / "rgb_to_bgr.jpg"), bgr_frame)
            print("Saved: rgb_to_bgr.jpg (RGB->BGR)")
            
            # Save with BGR->RGB conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(str(test_dir / "bgr_to_rgb.jpg"), rgb_frame)
            print("Saved: bgr_to_rgb.jpg (BGR->RGB)")
            
            print(f"\nTest images saved to {test_dir}/")
            print("Compare the images to see which has correct colors:")
            print("1. no_conversion.jpg - raw frame data")
            print("2. rgb_to_bgr.jpg - assuming input is RGB") 
            print("3. bgr_to_rgb.jpg - assuming input is BGR")
            
        else:
            print("Frame is not 3-channel color image")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera_manager.stop()

if __name__ == "__main__":
    test_color_formats()