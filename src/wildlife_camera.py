#!/usr/bin/env python3

import asyncio
import time
from pathlib import Path
from config import Config
from motion_detector import MotionDetector
from camera_manager import CameraManager
from telegram_service import TelegramService

class WildlifeCamera:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Initialize components
        self.camera = CameraManager(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.telegram_service = TelegramService(self.config)
        
        # Create data directory
        self.config.data_dir.mkdir(exist_ok=True)
        
        # State variables
        self.last_frame_time = 0
        self.last_detection_time = 0

    def cleanup_old_images(self):
        """Delete oldest images if we exceed the maximum limit"""
        try:
            image_files = sorted(
                list(self.config.storage.data_dir.glob(f"{self.config.storage.image_prefix}*.jpg")),
                key=lambda x: x.stat().st_mtime
            )
            
            if len(image_files) > self.config.performance.max_images:
                files_to_delete = image_files[:(len(image_files) - self.config.performance.max_images)]
                for file in files_to_delete:
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
        except Exception as e:
            print(f"Error in cleanup: {e}")

    async def send_telegram_message(self, image_path):
        """Send photo to Telegram channel"""
        caption = f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        return await self.telegram_service.send_photo_with_caption(image_path, caption)

    async def run(self):
        """Main loop for wildlife camera"""
        print("Wildlife Camera is running...")
        print(f"Motion detection parameters:")
        print(f"- Frame rate: {1/self.config.motion.frame_interval:.1f} FPS")
        print(f"- Motion threshold: {self.config.motion.motion_threshold}")
        print(f"- Minimum contour area: {self.config.motion.min_contour_area}")
        print(f"- Cooldown period: {self.config.performance.cooldown_period}s")
        print(f"- Maximum stored images: {self.config.performance.max_images}")
        print(f"- Consecutive detections required: {self.config.motion.consecutive_detections_required}")
        
        # Initial cleanup
        self.cleanup_old_images()
        
        try:
            while True:
                try:
                    current_time = time.time()
                    
                    # Frame rate control
                    if current_time - self.last_frame_time < self.config.motion.frame_interval:
                        await asyncio.sleep(self.config.performance.idle_sleep)
                        continue
                    
                    # Cooldown period check
                    if current_time - self.last_detection_time < self.config.performance.cooldown_period:
                        await asyncio.sleep(self.config.performance.cooldown_sleep)
                        continue
                    
                    self.last_frame_time = current_time
                    
                    # Capture and process frame
                    frame = self.camera.capture_motion_frame()
                    if frame is not None:
                        motion_result = self.motion_detector.detect(frame)
                        if motion_result.motion_detected:
                            print("Motion detected in central region!")
                            self.last_detection_time = current_time
                            
                            # Capture and send photo
                            image_path = self.camera.capture_and_save_photo()
                            if image_path:
                                await self.send_telegram_message(image_path)
                                self.cleanup_old_images()
                    
                    await asyncio.sleep(self.config.performance.idle_sleep)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    await asyncio.sleep(self.config.performance.error_sleep)
                    
        finally:
            # Cleanup on exit
            self.camera.stop()

if __name__ == "__main__":
    camera = WildlifeCamera()
    asyncio.run(camera.run()) 