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

    def reset_background_model(self):
        """Reset the background subtractor model to prevent false positives after detection."""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.motion.background_history,
            varThreshold=self.config.motion.background_threshold,
            detectShadows=False
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

    def _calculate_color_variance(self, bgr_frame, motion_mask):
        """
        Calculate color variance in the motion region.
        Higher variance indicates varied colors (animals), lower indicates uniform color (leaves).

        Args:
            bgr_frame: BGR color frame
            motion_mask: Binary mask of motion regions (0 or 255)

        Returns:
            Color variance score (higher = more color variety)
        """
        # Extract pixels in the motion region
        motion_pixels = bgr_frame[motion_mask > 0]

        if len(motion_pixels) == 0:
            return 0.0

        # Calculate variance in each color channel
        b_var = np.var(motion_pixels[:, 0])
        g_var = np.var(motion_pixels[:, 1])
        r_var = np.var(motion_pixels[:, 2])

        # Total color variance (sum of variances across channels)
        total_variance = b_var + g_var + r_var

        return float(total_variance)

    def detect(self, frame) -> MotionResult:
        """Detect motion in frame (RGB or grayscale) with color filtering and central region focus"""
        try:
            if frame is None:
                return MotionResult(motion_detected=False, motion_area=0)

            # Determine if frame is RGB or grayscale
            is_color = len(frame.shape) == 3 and frame.shape[2] == 3

            # Convert to grayscale for background subtraction
            if is_color:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.astype('uint8')

            # Apply Gaussian blur to reduce camera noise
            if self.config.motion.blur_kernel_size > 0:
                gray = cv2.GaussianBlur(gray, (self.config.motion.blur_kernel_size, self.config.motion.blur_kernel_size), 0)

            # Apply background subtraction and threshold
            fgmask = self.background_subtractor.apply(gray)
            # Increased threshold to 50 to ignore minor pixel fluctuations
            _, thresh = cv2.threshold(
                fgmask,
                50,  # Higher threshold to filter camera noise
                255,
                cv2.THRESH_BINARY
            )

            # Apply morphological operations to remove noise
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Fill small holes

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
                # Color variance filtering (if enabled and color frame available)
                passes_color_filter = True
                if is_color and self.config.motion.enable_color_filtering:
                    # Calculate color variance in the motion region
                    # Create a mask of the motion region
                    motion_mask = (thresh > 0).astype(np.uint8)

                    # Calculate color variance within the motion region
                    color_variance = self._calculate_color_variance(frame, motion_mask)

                    # Filter out low color variance (uniform color = leaves/inanimate objects)
                    if color_variance < self.config.motion.min_color_variance:
                        passes_color_filter = False

                if not passes_color_filter:
                    # Motion detected but filtered out due to low color variance
                    return MotionResult(
                        motion_detected=False,
                        motion_area=motion_area,
                        detection_confidence=0.3,  # Low confidence - filtered
                        center_x=center_x,
                        center_y=center_y,
                        contour_count=len(contours)
                    )

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

    def reset_background_model(self):
        """Reset the background subtractor model to prevent false positives after detection."""
        self._implementation.reset_background_model()