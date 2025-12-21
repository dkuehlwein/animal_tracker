import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from config import Config


class MotionDetectionError(Exception):
    """Base exception for motion detection errors."""
    pass


@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    motion_detected: bool
    motion_area: int
    detection_confidence: float = 0.0
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    contour_count: int = 0

class BaseMotionDetector:
    """Base motion detector implementation."""
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.motion.background_history,
            varThreshold=config.motion.background_threshold,
            detectShadows=False
        )
        
        # Create central region mask for new resolution
        self.central_region_mask = self._create_central_region_mask(
            config.camera.motion_detection_resolution[::-1]  # height, width
        )
        
        self.consecutive_detections = 0

    def _create_central_region_mask(self, shape):
        """Create a weighted mask that emphasizes the central region"""
        height, width = shape
        center_y, center_x = height // 2, width // 2
        Y, X = np.ogrid[:height, :width]
        
        # Calculate normalized distances from center
        dist_from_center = np.sqrt(
            (X - center_x)**2 / (width/2)**2 + 
            (Y - center_y)**2 / (height/2)**2
        )
        
        # Create weight mask
        weight_range = self.config.motion.center_weight - self.config.motion.edge_weight
        return np.clip(
            self.config.motion.center_weight - weight_range * dist_from_center,
            self.config.motion.edge_weight,
            self.config.motion.center_weight
        )

    def detect(self, frame) -> MotionResult:
        """Detect motion in YUV420 frame with central region focus"""
        try:
            # Frame is already Y channel (grayscale) from YUV420 format
            # No conversion needed - frame is already grayscale luminance
            if frame is None:
                return MotionResult(motion_detected=False, motion_area=0)
                
            gray = frame.astype('uint8')  # Ensure proper data type
            
            # Apply background subtraction and threshold
            fgmask = self.background_subtractor.apply(gray)
            # Use a low threshold (25) to detect pixel changes
            # motion_threshold is used later for total area
            _, thresh = cv2.threshold(
                fgmask, 
                25,  # Low threshold to detect any pixel-level changes
                255,
                cv2.THRESH_BINARY
            )
            
            # Apply central region weighting
            weighted_thresh = cv2.multiply(
                thresh.astype(np.float32),
                self.central_region_mask,
                dtype=cv2.CV_32F
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                weighted_thresh.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Check contours efficiently with Pi Zero optimized thresholds
            motion_area = 0
            significant_motion_detected = False
            center_x, center_y = None, None
            
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                motion_area += contour_area
                
                if contour_area > self.config.motion.min_contour_area:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        # Calculate centroid
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if centroid is in central region
                        h, w = gray.shape
                        min_x = w * self.config.motion.central_region_bounds[0]
                        max_x = w * self.config.motion.central_region_bounds[1]
                        min_y = h * self.config.motion.central_region_bounds[0]
                        max_y = h * self.config.motion.central_region_bounds[1]
                        
                        if min_x < cx < max_x and min_y < cy < max_y:
                            significant_motion_detected = True
                            center_x, center_y = cx, cy
                            break
            
            # Check if total motion area exceeds Pi Zero threshold
            if significant_motion_detected and motion_area >= self.config.motion.motion_threshold:
                self.consecutive_detections += 1
                if self.consecutive_detections >= self.config.motion.consecutive_detections_required:
                    self.consecutive_detections = 0
                    return MotionResult(
                        motion_detected=True,
                        motion_area=motion_area,
                        detection_confidence=1.0,
                        center_x=center_x,
                        center_y=center_y,
                        contour_count=len(contours)
                    )
                return MotionResult(
                    motion_detected=False,
                    motion_area=motion_area,
                    detection_confidence=0.5,
                    center_x=center_x,
                    center_y=center_y,
                    contour_count=len(contours)
                )
            
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            return MotionResult(
                motion_detected=False,
                motion_area=0,
                contour_count=len(contours)
            )
            
        except Exception as e:
            raise MotionDetectionError(f"Error in motion detection: {e}") from e


class WeightedMotionDetector(BaseMotionDetector):
    """Enhanced motion detector with configurable weighting strategies."""
    
    def __init__(self, config: Config, weight_strategy: str = "radial"):
        super().__init__(config)
        self.weight_strategy = weight_strategy
        self._update_weight_mask()
    
    def _update_weight_mask(self):
        """Update weight mask based on strategy."""
        if self.weight_strategy == "radial":
            # Use existing radial weighting
            pass
        elif self.weight_strategy == "rectangular":
            # Create rectangular central region weighting
            height, width = self.config.camera.motion_detection_resolution[::-1]
            mask = np.full((height, width), self.config.motion.edge_weight, dtype=np.float32)
            
            # Define central rectangle
            h_start = int(height * self.config.motion.central_region_bounds[0])
            h_end = int(height * self.config.motion.central_region_bounds[1])
            w_start = int(width * self.config.motion.central_region_bounds[0])
            w_end = int(width * self.config.motion.central_region_bounds[1])
            
            mask[h_start:h_end, w_start:w_end] = self.config.motion.center_weight
            self.central_region_mask = mask


class MotionDetector:
    """Main motion detector facade that uses WeightedMotionDetector as implementation."""

    def __init__(self, config: Config):
        self.config = config
        self._implementation = WeightedMotionDetector(config)

    def detect(self, frame) -> MotionResult:
        """Detect motion using the underlying implementation."""
        return self._implementation.detect(frame)