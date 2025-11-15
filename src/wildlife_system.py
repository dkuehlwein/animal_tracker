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
    Unified wildlife detection system with SpeciesNet-powered species identification.
    Combines motion detection, species identification, database logging, and Telegram notifications.
    """

    def __init__(self):
        # Load configuration
        self.config = Config()

        # Initialize all components
        self.camera = CameraManager(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.telegram_service = TelegramService(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.file_manager = FileManager(self.config)
        self.database = DatabaseManager(self.config)
        self.species_identifier = SpeciesIdentifier(self.config)

        # Setup
        self.file_manager.ensure_directories()
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
        """Process a detection with species identification and database logging"""
        timestamp = datetime.now()

        try:
            # Species identification with performance timing
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
        """Send notification with species identification info"""
        # Send text notification with species info
        await self.telegram_service.send_detection_notification(
            species_result, motion_area, timestamp
        )
    
    
    async def run(self):
        """Main loop for wildlife detection system"""
        print("Wildlife Detection System with SpeciesNet is running...")
        print(f"Motion detection parameters:")
        print(f"- Motion resolution: {self.config.camera.motion_detection_resolution} ({self.config.camera.motion_detection_format})")
        print(f"- Motion threshold: {self.config.motion.motion_threshold} pixels")
        print(f"- Frame interval: {self.config.motion.frame_interval}s ({1/self.config.motion.frame_interval:.1f} FPS)")
        print(f"- Cooldown period: {self.config.performance.cooldown_period}s")
        print(f"- Maximum stored images: {self.config.performance.max_images}")
        print(f"- Consecutive detections required: {self.config.motion.consecutive_detections_required}")
        print(f"Species identification:")
        print(f"- Model: {self.config.species.model_version}")
        print(f"- Location: {self.config.species.country_code}/{self.config.species.admin1_region}")
        print(f"- Unknown threshold: {self.config.species.unknown_species_threshold}")

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

                    # Memory check
                    if self.system_monitor.should_skip_processing():
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

                            # Log system status after large detections
                            if motion_area > self.config.motion.motion_threshold * 2:
                                self.system_monitor.log_system_status()

                        # Force cleanup after detection processing
                        self.system_monitor.memory_manager.force_cleanup()
                    
                    await asyncio.sleep(self.config.performance.idle_sleep)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    # Force cleanup on error
                    self.system_monitor.memory_manager.force_cleanup()
                    await asyncio.sleep(self.config.performance.error_sleep)
                    
        finally:
            # Cleanup on exit
            print("Cleaning up resources...")
            self.camera.stop()




if __name__ == "__main__":
    system = WildlifeSystem()
    asyncio.run(system.run())