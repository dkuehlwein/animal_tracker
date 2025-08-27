import cv2
import numpy as np
from config import Config

class MotionDetector:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.background_history,
            varThreshold=config.background_threshold,
            detectShadows=False
        )
        
        # Create central region mask for new resolution
        self.central_region_mask = self._create_central_region_mask(
            config.motion_detection_resolution[::-1]  # height, width
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
        weight_range = self.config.center_weight - self.config.edge_weight
        return np.clip(
            self.config.center_weight - weight_range * dist_from_center,
            self.config.edge_weight,
            self.config.center_weight
        )

    def detect(self, frame):
        """Detect motion in YUV420 frame with central region focus"""
        try:
            # Frame is already Y channel (grayscale) from YUV420 format
            # No conversion needed - frame is already grayscale luminance
            if frame is None:
                return False
                
            gray = frame.astype('uint8')  # Ensure proper data type
            
            # Apply background subtraction and threshold
            fgmask = self.background_subtractor.apply(gray)
            _, thresh = cv2.threshold(
                fgmask, 
                self.config.motion_threshold,
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
            
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                motion_area += contour_area
                
                if contour_area > self.config.min_contour_area:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        # Calculate centroid
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if centroid is in central region
                        h, w = gray.shape
                        min_x = w * self.config.central_region_bounds[0]
                        max_x = w * self.config.central_region_bounds[1]
                        min_y = h * self.config.central_region_bounds[0]
                        max_y = h * self.config.central_region_bounds[1]
                        
                        if min_x < cx < max_x and min_y < cy < max_y:
                            significant_motion_detected = True
                            break
            
            # Check if total motion area exceeds Pi Zero threshold
            if significant_motion_detected and motion_area >= self.config.motion_threshold:
                self.consecutive_detections += 1
                if self.consecutive_detections >= self.config.consecutive_detections_required:
                    self.consecutive_detections = 0
                    return True, motion_area  # Return motion area for database logging
                return False, motion_area
            
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            return False, 0  # Return consistent format
            
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return False, 0 