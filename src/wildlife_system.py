#!/usr/bin/env python3
"""
Unified Wildlife Detection System.
Combines motion detection, species identification, database logging, and Telegram notifications.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from config import Config
from motion_detector import MotionDetector
from camera_manager import CameraManager
from database_manager import DatabaseManager
from species_identifier import SpeciesIdentifier
from telegram_service import TelegramService
from utils import SystemMonitor, PerformanceTimer, FileManager

logger = logging.getLogger(__name__)


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

        # Thread pool for blocking operations (camera, species ID)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wildlife")

        # Setup
        self.file_manager.ensure_directories()
        self.telegram_service.set_database_reference(self.database)

        # State variables
        self.last_frame_time = 0
        self.last_detection_time = 0
    
    def cleanup_old_images(self):
        """Delete oldest images if we exceed the maximum limit - delegates to FileManager"""
        try:
            self.file_manager.cleanup_old_images()
        except Exception as e:
            logger.error(f"Error in cleanup: {e}", exc_info=True)
    
    def process_detection(self, image_path: Path, motion_area: int) -> tuple:
        """Process a detection with two-stage species identification and database logging"""
        timestamp = datetime.now()

        try:
            # Two-stage species identification with performance timing
            with PerformanceTimer("Two-stage species identification"):
                species_result = self.species_identifier.identify_species(image_path)

            # Log detection information
            if species_result.detection_result:
                det = species_result.detection_result
                logger.info(f"Stage 1 - MegaDetector: Found {det.detection_count} animals "
                            f"({det.processing_time:.2f}s)")
                if det.animals_detected:
                    logger.info(f"Stage 2 - Classifier: {species_result.species_name} "
                                f"(confidence: {species_result.confidence:.2f})")
                else:
                    logger.info("Stage 2 - Skipped (no animals detected)")
            else:
                # Legacy path without detection info
                logger.info(f"Identified: {species_result.species_name} "
                            f"(confidence: {species_result.confidence:.2f})")

            # Log to database
            detection_id = self.database.log_detection(
                image_path=image_path,
                motion_area=motion_area,
                species_name=species_result.species_name,
                confidence_score=species_result.confidence,
                processing_time=species_result.processing_time,
                api_success=species_result.api_success
            )

            logger.info(f"Detection {detection_id} logged: {species_result.species_name} "
                        f"(total time: {species_result.processing_time:.2f}s, motion: {motion_area} pixels)")

            # Convert IdentificationResult to dict for compatibility
            result_dict = {
                'species_name': species_result.species_name,
                'confidence': species_result.confidence,
                'api_success': species_result.api_success,
                'processing_time': species_result.processing_time,
                'fallback_reason': species_result.fallback_reason,
                'animals_detected': species_result.animals_detected,
                'detection_count': species_result.detection_result.detection_count if species_result.detection_result else 0
            }

            return result_dict, timestamp

        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True)
            # Return fallback result
            return {
                'species_name': 'Unknown species',
                'confidence': 0.0,
                'api_success': False,
                'processing_time': 0.0,
                'fallback_reason': f'Processing error: {e}',
                'animals_detected': False,
                'detection_count': 0
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
        logger.info("Wildlife Detection System with SpeciesNet is running...")
        logger.info("Motion detection parameters:")
        logger.info(f"- Motion resolution: {self.config.camera.motion_detection_resolution} ({self.config.camera.motion_detection_format})")
        logger.info(f"- Motion threshold: {self.config.motion.motion_threshold} pixels")
        logger.info(f"- Frame interval: {self.config.motion.frame_interval}s ({1/self.config.motion.frame_interval:.1f} FPS)")
        logger.info(f"- Cooldown period: {self.config.performance.cooldown_period}s")
        logger.info(f"- Maximum stored images: {self.config.performance.max_images}")
        logger.info(f"- Consecutive detections required: {self.config.motion.consecutive_detections_required}")
        logger.info("Species identification (Two-stage pipeline):")
        logger.info(f"- Stage 1: MegaDetector (min confidence: {self.config.species.min_detection_confidence})")
        logger.info(f"- Stage 2: SpeciesNet classifier (model: {self.config.species.model_version})")
        logger.info(f"- Location filtering: {self.config.species.country_code}/{self.config.species.admin1_region}")
        logger.info(f"- Unknown threshold: {self.config.species.unknown_species_threshold}")

        # Log initial system status
        self.system_monitor.log_system_status()
        
        # Initial cleanup
        self.file_manager.cleanup_old_images()
        
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

                    # Capture and process frame (run in executor to avoid blocking)
                    loop = asyncio.get_event_loop()
                    frame = await loop.run_in_executor(self.executor, self.camera.capture_motion_frame)

                    if frame is not None:
                        motion_result = self.motion_detector.detect(frame)
                        motion_detected = motion_result.motion_detected
                        motion_area = motion_result.motion_area
                    else:
                        motion_detected, motion_area = False, 0

                    if motion_detected:
                        logger.info(f"Motion detected in central region! Area: {motion_area} pixels")
                        self.last_detection_time = current_time

                        # Capture high-resolution photo (run in executor)
                        with PerformanceTimer("High-res capture"):
                            image_path = await loop.run_in_executor(
                                self.executor,
                                self.camera.capture_and_save_photo
                            )

                        if image_path:
                            # Process detection (species ID + database logging)
                            # Run in executor to avoid blocking event loop during 15-20s SpeciesNet inference
                            species_result, timestamp = await loop.run_in_executor(
                                self.executor,
                                self.process_detection,
                                image_path,
                                motion_area
                            )

                            # Send notification
                            await self.send_notification(species_result, motion_area, timestamp, image_path)

                            # Cleanup old images (now only when needed, not in hot path)
                            # Only cleanup after successful detection to avoid overhead
                            await loop.run_in_executor(self.executor, self.cleanup_old_images)

                            # Log system status after large detections
                            if motion_area > self.config.motion.motion_threshold * 2:
                                self.system_monitor.log_system_status()

                        # Force cleanup after detection processing
                        self.system_monitor.memory_manager.force_cleanup()
                    
                    await asyncio.sleep(self.config.performance.idle_sleep)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    # Force cleanup on error
                    self.system_monitor.memory_manager.force_cleanup()
                    await asyncio.sleep(self.config.performance.error_sleep)
                    
        finally:
            # Cleanup on exit
            logger.info("Cleaning up resources...")
            self.camera.stop()
            self.executor.shutdown(wait=True, cancel_futures=True)




if __name__ == "__main__":
    system = WildlifeSystem()
    asyncio.run(system.run())