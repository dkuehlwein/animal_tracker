#!/usr/bin/env python3

import asyncio
import telegram
import time
from datetime import datetime
from pathlib import Path

from config import Config
from motion_detector import MotionDetector
from camera_manager import CameraManager
from database_manager import DatabaseManager
from species_identifier import SpeciesIdentifier
from utils import SystemMonitor, PerformanceTimer, FileManager

class WildlifeDetector:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Initialize system monitoring
        self.system_monitor = SystemMonitor(self.config)
        self.file_manager = FileManager(self.config)
        
        # Ensure directories exist
        self.file_manager.ensure_directories()
        
        # Initialize components
        self.camera = CameraManager(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.database = DatabaseManager(self.config)
        self.species_identifier = SpeciesIdentifier(self.config)
        self.bot = telegram.Bot(token=self.config.telegram_token)
        
        # State variables
        self.last_frame_time = 0
        self.last_detection_time = 0

    async def send_telegram_notification(self, species_result, motion_area, timestamp):
        """Send species text notification to Telegram (not photos)"""
        try:
            species_name = species_result['species_name']
            confidence = species_result['confidence']
            
            # Format timestamp
            time_str = timestamp.strftime("%H:%M")
            
            # Create message based on species identification
            if species_name == "Unknown species":
                if species_result.get('fallback_reason') == 'Mock implementation':
                    message = f"🔍 Unknown species detected at {time_str}\nGarden camera - Motion area: {motion_area:,} pixels"
                else:
                    message = f"⚠️ Motion detected at {time_str}\nSpecies identification unavailable"
            else:
                # Check if this is first sighting today
                is_first_today = self.database.is_first_detection_today(species_name)
                first_sighting_text = " - First sighting today!" if is_first_today else ""
                
                if confidence > 0.8:
                    emoji = "🦔" if "Hedgehog" in species_name else "🦊" if "Fox" in species_name else "🐿️" if "Squirrel" in species_name else "🐱" if "Cat" in species_name else "🐦" if "Bird" in species_name or "Robin" in species_name or "Blackbird" in species_name else "🔍"
                    message = f"{emoji} {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}%{first_sighting_text}"
                else:
                    message = f"🔍 Possible {species_name} detected at {time_str}\nConfidence: {confidence*100:.0f}% - Motion area: {motion_area:,} pixels"
            
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message
            )
            
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")

    def process_detection(self, image_path, motion_area):
        """Process a detection with species identification and database logging"""
        timestamp = datetime.now()
        
        try:
            # Species identification with performance timing
            with PerformanceTimer("Species identification") as timer:
                species_result = self.species_identifier.identify_species(image_path)
            
            # Log to database
            detection_id = self.database.log_detection(
                image_path=image_path,
                motion_area=motion_area,
                species_name=species_result['species_name'],
                confidence_score=species_result['confidence'],
                processing_time=species_result['processing_time'],
                api_success=species_result['api_success']
            )
            
            print(f"Detection {detection_id}: {species_result['species_name']} "
                  f"(confidence: {species_result['confidence']:.2f}, "
                  f"motion: {motion_area} pixels)")
            
            return species_result, timestamp
            
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

    async def run(self):
        """Main loop for wildlife detector with Pi Zero optimizations"""
        print("Wildlife Detector is running...")
        print(f"Motion detection parameters:")
        print(f"- Motion resolution: {self.config.motion_detection_resolution} ({self.config.motion_detection_format})")
        print(f"- Capture resolution: {self.config.api_capture_resolution}")
        print(f"- Motion threshold: {self.config.motion_threshold} pixels")
        print(f"- Frame interval: {self.config.frame_interval}s ({1/self.config.frame_interval:.1f} FPS)")
        print(f"- Cooldown period: {self.config.cooldown_period}s")
        print(f"- Maximum stored images: {self.config.max_images}")
        print(f"- Consecutive detections required: {self.config.consecutive_detections_required}")
        
        # Log initial system status
        self.system_monitor.log_system_status()
        
        # Initial cleanup
        self.file_manager.cleanup_old_images()
        
        try:
            while True:
                try:
                    current_time = time.time()
                    
                    # Frame rate control
                    if current_time - self.last_frame_time < self.config.frame_interval:
                        await asyncio.sleep(self.config.idle_sleep)
                        continue
                    
                    # Cooldown period check
                    if current_time - self.last_detection_time < self.config.cooldown_period:
                        await asyncio.sleep(self.config.cooldown_sleep)
                        continue
                    
                    # Memory check for Pi Zero
                    if self.system_monitor.should_skip_processing():
                        self.system_monitor.memory_manager.force_cleanup()
                        await asyncio.sleep(self.config.error_sleep)
                        continue
                    
                    self.last_frame_time = current_time
                    
                    # Capture and process frame
                    with PerformanceTimer("Motion detection") as timer:
                        frame = self.camera.capture_motion_frame()
                        if frame is not None:
                            motion_detected, motion_area = self.motion_detector.detect(frame)
                        else:
                            motion_detected, motion_area = False, 0
                    
                    if motion_detected:
                        print(f"Motion detected in central region! Area: {motion_area} pixels")
                        self.last_detection_time = current_time
                        
                        # Capture high-resolution photo for species identification
                        with PerformanceTimer("High-res capture") as timer:
                            image_path = self.camera.capture_photo()
                        
                        if image_path:
                            # Process detection (species ID + database logging)
                            species_result, timestamp = self.process_detection(image_path, motion_area)
                            
                            # Send Telegram notification
                            await self.send_telegram_notification(species_result, motion_area, timestamp)
                            
                            # Cleanup old images
                            self.file_manager.cleanup_old_images()
                            
                            # Log system status after processing
                            if motion_area > self.config.motion_threshold * 2:  # Only for significant detections
                                self.system_monitor.log_system_status()
                        
                        # Force cleanup after detection processing
                        self.system_monitor.memory_manager.force_cleanup()
                    
                    await asyncio.sleep(self.config.idle_sleep)
                    
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    # Force cleanup on error
                    self.system_monitor.memory_manager.force_cleanup()
                    await asyncio.sleep(self.config.error_sleep)
                    
        finally:
            # Cleanup on exit
            print("Cleaning up resources...")
            self.camera.cleanup()

if __name__ == "__main__":
    detector = WildlifeDetector()
    asyncio.run(detector.run())