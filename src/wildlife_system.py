#!/usr/bin/env python3
"""
Unified Wildlife Detection System.
Combines motion detection, species identification, database logging, and Telegram notifications.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config
from motion_detector import MotionDetector
from camera_manager import CameraManager
from database_manager import DatabaseManager
from species_identifier import SpeciesIdentifier
from telegram_service import TelegramService
from utils import SystemMonitor, PerformanceTimer, FileManager


class WildlifeSystemError(Exception):
    """Base exception for wildlife system errors."""
    pass


class WildlifeSystem:
    """
    Unified wildlife detection system with configurable modes:
    - simple_mode: Just motion detection and photo capture (like original wildlife_camera)
    - advanced_mode: Full species identification and database logging (like wildlife_detector)
    """
    
    def __init__(self, advanced_mode: bool = True):
        # Load configuration
        self.config = Config()
        self.advanced_mode = advanced_mode
        
        # Initialize core components (always needed)
        self.camera = CameraManager(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.telegram_service = TelegramService(self.config)
        
        # Initialize advanced components (only if needed)
        if self.advanced_mode:
            self.system_monitor = SystemMonitor(self.config)
            self.file_manager = FileManager(self.config)
            self.file_manager.ensure_directories()
            
            self.database = DatabaseManager(self.config)
            self.species_identifier = SpeciesIdentifier(self.config)
            self.telegram_service.set_database_reference(self.database)
        
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
    
    def process_detection(self, image_path: Path, motion_area: int) -> tuple:
        """Process a detection with optional species identification and database logging"""
        timestamp = datetime.now()
        
        if not self.advanced_mode:
            # Simple mode: just return basic info
            return {
                'species_name': 'Motion detected',
                'confidence': 1.0,
                'api_success': False,
                'processing_time': 0.0,
                'fallback_reason': 'Simple mode'
            }, timestamp
        
        try:
            # Advanced mode: species identification with performance timing
            with PerformanceTimer("Species identification"):
                species_result = self.species_identifier.identify_species(image_path)
            
            # Log to database
            detection_id = self.database.log_detection(
                image_path=image_path,
                motion_area=motion_area,
                species_name=species_result.species_name,
                confidence_score=species_result.confidence,
                processing_time=species_result.processing_time,
                api_success=species_result.api_success
            )
            
            print(f"Detection {detection_id}: {species_result.species_name} "
                  f"(confidence: {species_result.confidence:.2f}, "
                  f"motion: {motion_area} pixels)")
            
            # Convert IdentificationResult to dict for compatibility
            result_dict = {
                'species_name': species_result.species_name,
                'confidence': species_result.confidence,
                'api_success': species_result.api_success,
                'processing_time': species_result.processing_time,
                'fallback_reason': species_result.fallback_reason
            }
            
            return result_dict, timestamp
            
        except Exception as e:
            print(f"Error processing detection: {e}")
            # Return fallback result
            return {
                'species_name': 'Unknown species',
                'confidence': 0.0,
                'api_success': False,
                'processing_time': 0.0,
                'fallback_reason': f'Processing error: {e}'
            }, timestamp
    
    async def send_notification(self, species_result: dict, motion_area: int, 
                               timestamp: datetime, image_path: Optional[Path] = None):
        """Send appropriate notification based on mode and settings"""
        if self.advanced_mode:
            # Advanced mode: text notifications with species info
            await self.telegram_service.send_detection_notification(
                species_result, motion_area, timestamp
            )
        else:
            # Simple mode: photo notifications
            if image_path:
                caption = f"Motion detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                await self.telegram_service.send_photo_with_caption(image_path, caption)
    
    async def send_telegram_notification(self, species_result: dict, motion_area: int, timestamp: datetime):
        """Backward compatibility method for sending notifications"""
        return await self.send_notification(species_result, motion_area, timestamp)
    
    async def run(self):
        """Main loop for wildlife detection system"""
        mode_str = "Advanced" if self.advanced_mode else "Simple"
        print(f"Wildlife Detection System ({mode_str} Mode) is running...")
        print(f"Motion detection parameters:")
        print(f"- Motion resolution: {self.config.camera.motion_detection_resolution} ({self.config.camera.motion_detection_format})")
        print(f"- Motion threshold: {self.config.motion.motion_threshold} pixels")
        print(f"- Frame interval: {self.config.motion.frame_interval}s ({1/self.config.motion.frame_interval:.1f} FPS)")
        print(f"- Cooldown period: {self.config.performance.cooldown_period}s")
        print(f"- Maximum stored images: {self.config.performance.max_images}")
        print(f"- Consecutive detections required: {self.config.motion.consecutive_detections_required}")
        
        if self.advanced_mode:
            # Log initial system status
            self.system_monitor.log_system_status()
        
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
                    
                    # Memory check for advanced mode
                    if self.advanced_mode and self.system_monitor.should_skip_processing():
                        self.system_monitor.memory_manager.force_cleanup()
                        await asyncio.sleep(self.config.performance.error_sleep)
                        continue
                    
                    self.last_frame_time = current_time
                    
                    # Capture and process frame
                    frame = self.camera.capture_motion_frame()
                    if frame is not None:
                        motion_result = self.motion_detector.detect(frame)
                        motion_detected = motion_result.motion_detected
                        motion_area = motion_result.motion_area
                    else:
                        motion_detected, motion_area = False, 0
                    
                    if motion_detected:
                        print(f"Motion detected in central region! Area: {motion_area} pixels")
                        self.last_detection_time = current_time
                        
                        # Capture high-resolution photo
                        with PerformanceTimer("High-res capture"):
                            image_path = self.camera.capture_and_save_photo()
                        
                        if image_path:
                            # Process detection (species ID + database logging in advanced mode)
                            species_result, timestamp = self.process_detection(image_path, motion_area)
                            
                            # Send notification
                            await self.send_notification(species_result, motion_area, timestamp, image_path)
                            
                            # Cleanup old images
                            self.cleanup_old_images()
                            
                            # Log system status after processing (advanced mode only)
                            if (self.advanced_mode and 
                                motion_area > self.config.motion.motion_threshold * 2):
                                self.system_monitor.log_system_status()
                        
                        # Force cleanup after detection processing (advanced mode)
                        if self.advanced_mode:
                            self.system_monitor.memory_manager.force_cleanup()
                    
                    await asyncio.sleep(self.config.performance.idle_sleep)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    # Force cleanup on error (advanced mode)
                    if self.advanced_mode:
                        self.system_monitor.memory_manager.force_cleanup()
                    await asyncio.sleep(self.config.performance.error_sleep)
                    
        finally:
            # Cleanup on exit
            print("Cleaning up resources...")
            self.camera.stop()


# Backward compatibility classes
class WildlifeCamera(WildlifeSystem):
    """Backward compatibility: Simple mode wildlife system (like original wildlife_camera)"""
    def __init__(self):
        super().__init__(advanced_mode=False)


class WildlifeDetector(WildlifeSystem):
    """Backward compatibility: Advanced mode wildlife system (like original wildlife_detector)"""
    def __init__(self):
        super().__init__(advanced_mode=True)


if __name__ == "__main__":
    import sys
    
    # Determine mode from command line argument
    advanced_mode = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['simple', 'camera']:
        advanced_mode = False
    
    system = WildlifeSystem(advanced_mode=advanced_mode)
    asyncio.run(system.run())