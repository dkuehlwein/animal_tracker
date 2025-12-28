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
from notification_service import NotificationService
from resource_manager import SystemMonitor, StorageManager
from utils import PerformanceTimer, SunChecker, MotionVisualizer, SharpnessAnalyzer
from exceptions import WildlifeSystemError

logger = logging.getLogger(__name__)


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
        self.telegram_service = NotificationService(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.file_manager = StorageManager(self.config)
        self.database = DatabaseManager(self.config)
        self.species_identifier = SpeciesIdentifier(self.config)
        self.sun_checker = SunChecker(self.config)

        # Thread pool for blocking operations (camera, species ID)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wildlife")

        # Setup
        self.file_manager.ensure_directories()
        self.telegram_service.set_database_reference(self.database)

        # State variables
        self.last_frame_time = 0
        self.last_detection_time = 0
        self.last_status_log_time = 0
        self.last_motion_frame = None
        self.last_motion_result = None

        # Reference frame for burst selection (cached background when no motion)
        self.reference_frame = None
        self.reference_frame_time = 0
        self.reference_update_interval = 600  # Update every 10 minutes

        # Daylight state tracking for start/stop notifications
        self._was_daytime = None  # None = unknown, will be set on first check

    async def _check_daylight_transition(self) -> bool:
        """
        Check for daylight state transitions and send notifications.
        Returns True if currently daytime (detection should run).
        """
        is_daytime = self.sun_checker.is_daytime()
        sun_info = self.sun_checker.get_sun_info()

        # First check - initialize state without notification
        if self._was_daytime is None:
            self._was_daytime = is_daytime
            if is_daytime:
                logger.info("System started during daytime - detection active")
            else:
                logger.info("System started during nighttime - detection paused")
            return is_daytime

        # Transition from night to day - detection starting
        if not self._was_daytime and is_daytime:
            self._was_daytime = True
            logger.info("Sunrise transition - starting detection")
            await self.telegram_service.send_text_message(
                f"🌅 Good morning! Wildlife detection starting.\n"
                f"Sunrise: {sun_info['sunrise']}, Sunset: {sun_info['sunset']}"
            )
            return True

        # Transition from day to night - detection stopping
        if self._was_daytime and not is_daytime:
            self._was_daytime = False
            # Get today's detection stats
            today_count = self.database.get_daily_detections()
            logger.info(f"Sunset transition - stopping detection ({today_count} detections today)")
            await self.telegram_service.send_text_message(
                f"🌙 Good evening! Wildlife detection paused for the night.\n"
                f"Today's detections: {today_count}\n"
                f"Resuming at sunrise ({sun_info['sunrise']} tomorrow)"
            )
            return False

        self._was_daytime = is_daytime
        return is_daytime

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
    
    def _capture_and_select_best_frame(self) -> tuple:
        """
        Capture burst of frames, analyze sharpness, save best frame.
        Returns (image_path, sharpness_info_dict) or (None, None) on failure.
        """
        try:
            # Capture burst frames
            frames = self.camera.capture_burst_frames(
                count=self.config.performance.multi_frame_count,
                interval=self.config.performance.multi_frame_interval
            )
            
            if not frames:
                logger.error("Burst capture failed, no frames captured")
                return None, None
            
            # Analyze sharpness and select best frame (using cached reference if available)
            best_frame, selected_index, best_score, all_scores = SharpnessAnalyzer.select_sharpest_frame(
                frames,
                motion_aware=self.config.performance.motion_aware_selection,
                reference_frame=self.reference_frame
            )
            
            if best_frame is None:
                logger.error("Sharpness analysis failed")
                return None, None
            
            # Check if sharpness meets minimum threshold
            if best_score < self.config.performance.min_sharpness_threshold:
                logger.warning(f"Best frame sharpness ({best_score:.1f}) below threshold "
                             f"({self.config.performance.min_sharpness_threshold})")
            
            # Generate timestamped filename and save best frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config.storage.image_dir / f"{self.config.storage.image_prefix}{timestamp}.jpg"
            
            # Save the best frame
            import cv2
            success = cv2.imwrite(str(file_path), best_frame)
            
            if not success:
                logger.error(f"Failed to save best frame to {file_path}")
                return None, None
            
            logger.info(f"Burst capture complete: saved frame {selected_index + 1}/{len(frames)} "
                       f"(sharpness: {best_score:.1f})")
            
            # Build sharpness info dict for notification
            sharpness_info = {
                'sharpness_score': best_score,
                'selected_frame_index': selected_index,
                'frame_count': len(frames),
                'all_scores': all_scores,
                'meets_threshold': best_score >= self.config.performance.min_sharpness_threshold
            }
            
            return file_path, sharpness_info
        
        except Exception as e:
            logger.error(f"Error in burst capture and selection: {e}", exc_info=True)
            return None, None
    
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
                # Convert to int if it's a string
                if isinstance(cat, str):
                    cat = int(cat)
                conf = det.get('conf', 0.0)
                cat_name = category_names.get(cat, f'category {cat}')
                detected_items.append(f"{cat_name} ({conf:.1%})")
                if cat == 1:  # Animal category
                    animals_detected = True
        
        # Build base caption
        caption = ""
        
        # If animals were detected, show species identification
        if animals_detected:
            caption = f"🦌 {species_name}\nConfidence: {confidence:.1%}\nMotion area: {motion_area} px\n{time_str}"
        # Otherwise show what was detected
        elif detected_items:
            items_str = ", ".join(detected_items)
            caption = f"👁️ Motion detected\nFound: {items_str}\nNo animals above threshold\nMotion area: {motion_area} px\n{time_str}"
        else:
            caption = f"👁️ Motion detected\nNo objects identified by detector\nMotion area: {motion_area} px\n{time_str}"
        
        # Add sharpness info if available
        sharpness_info = species_result.get('sharpness_info')
        if sharpness_info:
            caption += f"\n\n📸 Image Quality:"
            caption += f"\nSharpness: {sharpness_info['sharpness_score']:.1f}"
            caption += f"\nSelected: Frame {sharpness_info['selected_frame_index'] + 1}/{sharpness_info['frame_count']}"
            if not sharpness_info['meets_threshold']:
                caption += f"\n⚠️ Below quality threshold ({self.config.performance.min_sharpness_threshold:.0f})"
        
        return caption
    
    async def send_notification(self, species_result: dict, motion_area: int,
                               timestamp: datetime, image_path: Optional[Path] = None,
                               annotated_path: Optional[Path] = None):
        """Send notification with species identification info and motion visualization"""
        # Get system temperature
        temperature = self.system_monitor.get_cpu_temperature()
        
        if image_path and image_path.exists():
            caption = self._build_caption(species_result, motion_area, timestamp)
            
            if temperature:
                caption += f"\n🌡️ {temperature:.1f}°C"
                
            # If we have both original and annotated images, send as media group
            if annotated_path and annotated_path.exists():
                await self.telegram_service.send_media_group(
                    [image_path, annotated_path],
                    caption
                )
            else:
                # Fallback to single image
                await self.telegram_service.send_photo_with_caption(image_path, caption)
        else:
            await self.telegram_service.send_detection_notification(
                species_result, motion_area, timestamp, temperature
            )
    
    
    async def run(self):
        """Main loop for wildlife detection system"""
        logger.info("Wildlife Detection System with SpeciesNet is running...")
        logger.info("Motion detection parameters:")
        logger.info(f"- Motion resolution: {self.config.camera.motion_detection_resolution} ({self.config.camera.motion_detection_format})")
        logger.info(f"- Motion threshold: {self.config.motion.motion_threshold} pixels")
        logger.info(f"- Frame interval: {self.config.motion.frame_interval}s ({1/self.config.motion.frame_interval:.1f} FPS)")
        logger.info(f"- Cooldown period: {self.config.performance.cooldown_period}s")
        logger.info(f"- Capture delay: {self.config.performance.capture_delay}s (wait for animal to settle)")
        logger.info(f"- Maximum stored images: {self.config.performance.max_images}")
        logger.info(f"- Consecutive detections required: {self.config.motion.consecutive_detections_required}")
        logger.info("Species identification (Two-stage pipeline):")
        logger.info(f"- Stage 1: MegaDetector (min confidence: {self.config.species.min_detection_confidence})")
        logger.info(f"- Stage 2: SpeciesNet classifier (model: {self.config.species.model_version})")
        logger.info(f"- Location filtering: {self.config.species.country_code}/{self.config.species.admin1_region}")
        logger.info(f"- Unknown threshold: {self.config.species.unknown_species_threshold}")
        
        # Log multi-frame capture settings
        logger.info("Image capture:")
        if self.config.performance.enable_multi_frame:
            logger.info(f"- Multi-frame burst capture: ENABLED")
            logger.info(f"  • Frames per burst: {self.config.performance.multi_frame_count}")
            logger.info(f"  • Interval: {self.config.performance.multi_frame_interval}s")
            logger.info(f"  • Total burst time: ~{self.config.performance.multi_frame_count * self.config.performance.multi_frame_interval:.1f}s")
            logger.info(f"  • Sharpness threshold: {self.config.performance.min_sharpness_threshold}")
            logger.info(f"  • Selection: Best frame by Laplacian variance")
        else:
            logger.info(f"- Multi-frame burst capture: DISABLED (single frame mode)")

        # Log daylight settings
        if self.config.performance.daylight_only:
            sun_info = self.sun_checker.get_sun_info()
            logger.info(f"Daylight tracking enabled:")
            logger.info(f"- Sunrise: {sun_info['sunrise']}, Sunset: {sun_info['sunset']}")
            logger.info(f"- Currently: {'Daytime' if sun_info['is_daytime'] else 'Nighttime'}")
        else:
            logger.info("24/7 tracking enabled (daylight checking disabled)")

        # Log initial system status
        self.system_monitor.log_system_status()
        
        # Initial cleanup
        self.file_manager.cleanup_old_images()
        
        # Start camera session
        logger.info("Initializing camera...")
        with self.camera.camera_session():
            logger.info("Camera initialized successfully!")

            # Capture initial reference frame for burst selection
            try:
                ref_frame = self.camera.capture_high_res_frame()
                if ref_frame is not None:
                    self.reference_frame = ref_frame.copy()
                    self.reference_frame_time = time.time()
                    logger.info("Captured initial reference frame for burst selection")
            except Exception as e:
                logger.warning(f"Failed to capture initial reference frame: {e}")
            
            try:
                while True:
                    try:
                        current_time = time.time()

                        # Check daylight if enabled (with transition notifications)
                        if self.config.performance.daylight_only:
                            is_daytime = await self._check_daylight_transition()
                            if not is_daytime:
                                # Sleep longer during nighttime to save resources
                                await asyncio.sleep(60)  # Check every minute at night
                                continue
                        
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

                            # Store frame and result for later annotation
                            self.last_motion_frame = frame.copy()
                            self.last_motion_result = motion_result

                            # Cache reference frame when no motion detected (for burst frame selection)
                            # Update every 10 minutes to account for lighting changes
                            if not motion_detected and motion_area == 0:
                                time_since_ref_update = current_time - self.reference_frame_time
                                if time_since_ref_update >= self.reference_update_interval:
                                    # Capture high-res reference frame
                                    ref_frame = self.camera.capture_high_res_frame()
                                    if ref_frame is not None:
                                        self.reference_frame = ref_frame.copy()
                                        self.reference_frame_time = current_time
                                        logger.info(f"Updated reference frame for burst selection (age: {time_since_ref_update:.0f}s)")

                            # Log motion status periodically (every 5 seconds)
                            if current_time - self.last_status_log_time >= 5.0:
                                self.last_status_log_time = current_time
                                logger.info(f"Monitoring... motion_area={motion_area} (threshold={self.config.motion.motion_threshold})")
                        else:
                            motion_detected, motion_area = False, 0
                            logger.warning("Failed to capture frame")

                        if motion_detected:
                            logger.info(f"Motion detected in central region! Area: {motion_area} pixels")
                            self.last_detection_time = current_time

                            # Multi-frame or single-frame capture based on config
                            sharpness_info = None
                            if self.config.performance.enable_multi_frame:
                                with PerformanceTimer("Multi-frame burst capture"):
                                    image_path, sharpness_info = await loop.run_in_executor(
                                        self.executor,
                                        self._capture_and_select_best_frame
                                    )
                            else:
                                # Fallback to single-frame capture
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
                                
                                # Add sharpness info to species result for notification
                                if sharpness_info:
                                    species_result['sharpness_info'] = sharpness_info

                                # Create annotated image showing motion detection regions
                                annotated_path = None
                                if self.last_motion_frame is not None and self.last_motion_result is not None:
                                    annotated_path = await loop.run_in_executor(
                                        self.executor,
                                        MotionVisualizer.create_annotated_image,
                                        image_path,
                                        self.last_motion_frame,
                                        self.config,
                                        self.last_motion_result
                                    )

                                # Send notification with both original and annotated images
                                await self.send_notification(species_result, motion_area, timestamp,
                                                            image_path, annotated_path)

                                # Cleanup old images (now only when needed, not in hot path)
                                # Only cleanup after successful detection to avoid overhead
                                await loop.run_in_executor(self.executor, self.cleanup_old_images)

                                # Log system status after large detections
                                if motion_area > self.config.motion.motion_threshold * 2:
                                    self.system_monitor.log_system_status()

                            # Reset background model to prevent false positives from lingering motion
                            self.motion_detector.reset_background_model()
                            logger.info("Background model reset to prevent false alarms")

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