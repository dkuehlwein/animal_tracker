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
                'detection_count': species_result.detection_result.detection_count if species_result.detection_result else 0,
                'detection_result': species_result.detection_result  # Include full detection info
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
    
    def _extract_species_name(self, species_name_raw: str) -> str:
        """Extract human-readable name from taxonomic path."""
        # Format: UUID;class;order;family;genus;species;common_name
        if ';' in species_name_raw:
            parts = species_name_raw.split(';')
            species_name = parts[-1].strip() if parts[-1].strip() else 'Unknown species'
            return species_name.title()
        return species_name_raw
    
    def _build_caption(self, species_result: dict, motion_area: int, timestamp: datetime) -> str:
        """Build notification caption based on detection results."""
        species_name = self._extract_species_name(species_result.get('species_name', 'Unknown species'))
        confidence = species_result.get('confidence', 0.0)
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check what MegaDetector found
        detection_result = species_result.get('detection_result')
        detected_items = []
        animals_detected = False
        
        if detection_result and detection_result.detections:
            category_names = {1: 'animal', 2: 'person', 3: 'vehicle'}
            for det in detection_result.detections:
                cat = det.get('category')
                conf = det.get('conf', 0.0)
                cat_name = category_names.get(cat, f'unknown (category {cat})')
                detected_items.append(f"{cat_name} ({conf:.1%})")
                if cat == 1:  # Animal category
                    animals_detected = True
        
        # If animals were detected, show species identification
        if animals_detected:
            return f"🦌 {species_name}\nConfidence: {confidence:.1%}\nMotion area: {motion_area} px\n{time_str}"
        
        # Otherwise show what was detected
        if detected_items:
            items_str = ", ".join(detected_items)
            return f"👁️ Motion detected\nFound: {items_str}\nNo animals above threshold\nMotion area: {motion_area} px\n{time_str}"
        
        return f"👁️ Motion detected\nNo objects identified by detector\nMotion area: {motion_area} px\n{time_str}"
    
    async def send_notification(self, species_result: dict, motion_area: int,
                               timestamp: datetime, image_path: Optional[Path] = None):
        """Send notification with species identification info"""
        if image_path and image_path.exists():
            caption = self._build_caption(species_result, motion_area, timestamp)
            await self.telegram_service.send_photo_with_caption(image_path, caption)
        else:
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
        
        # Start camera session
        logger.info("Initializing camera...")
        with self.camera.camera_session():
            logger.info("Camera initialized successfully!")
            
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
                            # Log motion status periodically (every 5 seconds)
                            if int(current_time) % 5 == 0 and current_time - self.last_frame_time < 0.3:
                                logger.info(f"Monitoring... motion_area={motion_area} (threshold={self.config.motion.motion_threshold})")
                        else:
                            motion_detected, motion_area = False, 0
                            logger.warning("Failed to capture frame")

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
                self.executor.shutdown(wait=True, cancel_futures=True)




if __name__ == "__main__":
    # Setup logging - INFO for most, WARNING for picamera2
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Reduce picamera2 verbosity
    logging.getLogger('picamera2').setLevel(logging.WARNING)
    
    system = WildlifeSystem()
    asyncio.run(system.run())