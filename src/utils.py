"""
Utility classes for the wildlife detection system.

This module contains:
- PerformanceTimer: Timing utility for performance measurement
- MotionVisualizer: Create annotated images showing motion detection regions
- SharpnessAnalyzer: Image sharpness analysis for burst capture selection
- SunChecker: Daylight checking based on sunrise/sunset times
"""

import logging
import time
import zoneinfo
from datetime import datetime, date, timezone, time as dt_time

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from astral import LocationInfo
from astral.sun import sun

from config import Config

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Performance timing utility."""

    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def stop(self):
        """Stop timing and return duration."""
        if self.start_time is None:
            return 0.0

        self.end_time = time.time()
        duration = self.end_time - self.start_time
        return duration

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        duration = self.stop()
        if duration > 1.0:  # Log slow operations
            logger.info(f"{self.operation_name} took {duration:.2f}s")


class MotionVisualizer:
    """Create annotated images showing motion detection regions."""

    @staticmethod
    def create_annotated_image(image_path: Path, motion_frame, config, motion_result) -> Optional[Path]:
        """
        Create annotated version of image showing motion detection regions.

        Args:
            image_path: Path to the original high-res image
            motion_frame: The low-res grayscale frame used for motion detection
            config: System configuration
            motion_result: MotionResult object with detection details

        Returns:
            Path to the annotated image file
        """
        try:
            # Load the high-res image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None

            img_height, img_width = img.shape[:2]

            # Get motion detection parameters
            motion_width, motion_height = config.camera.motion_detection_resolution

            # Calculate scaling factors from motion frame to high-res image
            scale_x = img_width / motion_width
            scale_y = img_height / motion_height

            # Draw central region boundary (where motion is prioritized)
            bounds = config.motion.central_region_bounds
            center_x1 = int(img_width * bounds[0])
            center_x2 = int(img_width * bounds[1])
            center_y1 = int(img_height * bounds[0])
            center_y2 = int(img_height * bounds[1])

            # Draw semi-transparent overlay for central region
            overlay = img.copy()
            cv2.rectangle(overlay, (center_x1, center_y1), (center_x2, center_y2),
                         (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw the actual motion detection point if available
            if motion_result.center_x is not None and motion_result.center_y is not None:
                # Scale the detection point to high-res coordinates
                det_x = int(motion_result.center_x * scale_x)
                det_y = int(motion_result.center_y * scale_y)

                # Draw crosshair at detection point
                cv2.drawMarker(img, (det_x, det_y), (0, 0, 255),
                              markerType=cv2.MARKER_CROSS, markerSize=50, thickness=3)

                # Draw circle around detection point
                cv2.circle(img, (det_x, det_y), 30, (0, 0, 255), 3)

            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Motion info text
            text_lines = [
                f"Motion Area: {motion_result.motion_area} px",
                f"Threshold: {config.motion.motion_threshold} px",
                f"Contours: {motion_result.contour_count}",
                f"Confidence: {motion_result.detection_confidence:.1%}"
            ]

            # Add text with background for readability
            y_offset = 30
            for line in text_lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                # Draw black background rectangle
                cv2.rectangle(img, (5, y_offset - 25),
                            (text_size[0] + 15, y_offset + 5),
                            (0, 0, 0), -1)
                # Draw white text
                cv2.putText(img, line, (10, y_offset), font, font_scale,
                           (255, 255, 255), thickness)
                y_offset += 35

            # Add legend
            legend_y = img_height - 70
            cv2.rectangle(img, (5, legend_y - 25), (270, img_height - 10), (0, 0, 0), -1)
            cv2.putText(img, "Green: Central Region", (10, legend_y),
                       font, 0.5, (0, 255, 0), 1)
            cv2.putText(img, "Red: Motion Detection Point", (10, legend_y + 25),
                       font, 0.5, (0, 0, 255), 1)

            # Save annotated image with _annotated suffix
            annotated_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
            cv2.imwrite(str(annotated_path), img)

            logger.info(f"Created annotated image: {annotated_path}")
            return annotated_path

        except Exception as e:
            logger.error(f"Error creating annotated image: {e}", exc_info=True)
            return None


class SharpnessAnalyzer:
    """Analyze image sharpness using Laplacian variance for burst capture selection."""

    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher values indicate sharper images.

        Args:
            frame: BGR or grayscale image array

        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Calculate Laplacian variance (measures edge strength)
            # Higher variance = more edges = sharper image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            return float(variance)

        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0

    @staticmethod
    def calculate_difference_from_reference(frame: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate how different a frame is from a reference (background) frame.
        Higher score = more difference = more subject content visible.

        Args:
            frame: BGR or grayscale image
            reference: Reference (background) frame to compare against

        Returns:
            Difference score (mean absolute difference)
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if len(reference.shape) == 3:
                ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = reference

            # Resize reference to match frame if needed
            if gray.shape != ref_gray.shape:
                ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))

            # Calculate absolute difference
            diff = cv2.absdiff(gray, ref_gray)

            # Return mean difference (higher = more subject content)
            return float(np.mean(diff))

        except Exception as e:
            logger.error(f"Error calculating difference from reference: {e}")
            return 0.0

    @staticmethod
    def select_sharpest_frame(frames: list, motion_aware: bool = True,
                             reference_frame: np.ndarray = None) -> tuple:
        """
        Analyze multiple frames and select the best one using combined scoring.

        When motion_aware is True and reference_frame is provided:
        - Compares each frame to the cached background reference
        - Frame with biggest difference has most subject content visible
        - Combined with sharpness for final selection

        Args:
            frames: List of numpy arrays (BGR or grayscale images)
            motion_aware: If True, use reference-based scoring when available
            reference_frame: Cached background frame from when scene was empty

        Returns:
            Tuple of (best_frame, selected_index, best_score, all_scores)
        """
        if not frames:
            logger.warning("No frames provided for sharpness analysis")
            return None, -1, 0.0, []

        try:
            # Calculate sharpness for all frames
            sharpness_scores = [SharpnessAnalyzer.calculate_sharpness(frame) for frame in frames]

            # If no reference frame or motion_aware disabled, just use sharpness
            if not motion_aware or reference_frame is None:
                best_index = sharpness_scores.index(max(sharpness_scores))
                best_frame = frames[best_index]
                best_score = sharpness_scores[best_index]

                logger.info(f"Sharpness-only selection: scores={[f'{s:.1f}' for s in sharpness_scores]}, "
                           f"selected frame {best_index + 1}/{len(frames)} (score: {best_score:.1f})"
                           f"{' (no reference)' if motion_aware else ''}")

                return best_frame, best_index, best_score, sharpness_scores

            # Reference-based selection: compare each frame to cached background
            diff_scores = [
                SharpnessAnalyzer.calculate_difference_from_reference(frame, reference_frame)
                for frame in frames
            ]

            # Normalize scores to 0-1 range for combining
            max_sharpness = max(sharpness_scores) if max(sharpness_scores) > 0 else 1
            max_diff = max(diff_scores) if max(diff_scores) > 0 else 1

            normalized_sharpness = [s / max_sharpness for s in sharpness_scores]
            normalized_diff = [d / max_diff for d in diff_scores]

            # Combined score: 40% sharpness, 60% difference from background
            # Prioritize frames where subject is most visible
            combined_scores = [
                0.4 * ns + 0.6 * nd
                for ns, nd in zip(normalized_sharpness, normalized_diff)
            ]

            # Select frame with best combined score
            best_index = combined_scores.index(max(combined_scores))
            best_frame = frames[best_index]
            best_score = sharpness_scores[best_index]

            logger.info(f"Reference-based selection: "
                       f"sharpness={[f'{s:.1f}' for s in sharpness_scores]}, "
                       f"diff_from_bg={[f'{d:.1f}' for d in diff_scores]}, "
                       f"combined={[f'{c:.2f}' for c in combined_scores]}, "
                       f"selected frame {best_index + 1}/{len(frames)} "
                       f"(sharpness: {best_score:.1f}, diff: {diff_scores[best_index]:.1f})")

            return best_frame, best_index, best_score, sharpness_scores

        except Exception as e:
            logger.error(f"Error in sharpness analysis: {e}")
            # Return first frame as fallback
            return frames[0] if frames else None, 0, 0.0, []


class SunChecker:
    """Check if it's currently daytime based on sunrise/sunset times."""

    def __init__(self, config: Config):
        self.config = config
        self._local_tz = zoneinfo.ZoneInfo(config.location.timezone)
        self.location = LocationInfo(
            name="Location",
            region=config.location.timezone,
            timezone=config.location.timezone,
            latitude=config.location.latitude,
            longitude=config.location.longitude
        )
        self._last_check_date = None
        self._sunrise = None
        self._sunset = None
        logger.info(f"SunChecker initialized for location: "
                   f"{config.location.latitude:.4f}, {config.location.longitude:.4f} "
                   f"(timezone: {config.location.timezone})")

    def _update_sun_times(self):
        """Update sunrise/sunset times for today (handles DST automatically)."""
        try:
            today = date.today()
            if self._last_check_date != today:
                # astral returns timezone-aware UTC times
                s = sun(self.location.observer, date=today)
                self._sunrise = s['sunrise']
                self._sunset = s['sunset']
                self._last_check_date = today
                # Log in local time for readability (DST-aware via zoneinfo)
                sunrise_local = self._sunrise.astimezone(self._local_tz)
                sunset_local = self._sunset.astimezone(self._local_tz)
                logger.info(f"Sun times updated - Sunrise: {sunrise_local.strftime('%H:%M')}, "
                           f"Sunset: {sunset_local.strftime('%H:%M')} (local time)")
        except Exception as e:
            logger.error(f"Error calculating sun times: {e}")
            # Fallback to safe defaults (6am - 8pm in local timezone, DST-aware)
            now = datetime.now(self._local_tz)
            self._sunrise = datetime.combine(
                now.date(), dt_time(6, 0), tzinfo=self._local_tz
            )
            self._sunset = datetime.combine(
                now.date(), dt_time(20, 0), tzinfo=self._local_tz
            )

    def is_daytime(self) -> bool:
        """Check if it's currently daytime."""
        self._update_sun_times()

        now = datetime.now(timezone.utc)

        # Both sunrise/sunset are timezone-aware, comparison works correctly
        if self._sunrise and self._sunset:
            return self._sunrise <= now <= self._sunset

        # Fallback if times aren't set
        return True

    def get_sun_info(self) -> dict:
        """Get sunrise/sunset information in local time (DST-aware)."""
        self._update_sun_times()

        # Convert to local timezone for display (handles DST automatically)
        sunrise_local = self._sunrise.astimezone(self._local_tz) if self._sunrise else None
        sunset_local = self._sunset.astimezone(self._local_tz) if self._sunset else None

        return {
            'sunrise': sunrise_local.strftime('%H:%M') if sunrise_local else 'Unknown',
            'sunset': sunset_local.strftime('%H:%M') if sunset_local else 'Unknown',
            'is_daytime': self.is_daytime()
        }
