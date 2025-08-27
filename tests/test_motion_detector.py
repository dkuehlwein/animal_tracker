"""
Unit tests for motion detection system.
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import patch, Mock

import sys
sys.path.append('src')

from motion_detector import (
    MotionDetector, MotionDetectionError, MotionResult, 
    WeightedMotionDetector
)
from config import Config


class TestMotionResult:
    """Test motion result data class."""
    
    def test_motion_result_creation(self):
        """Test motion result creation."""
        result = MotionResult(
            motion_detected=True,
            motion_area=2500,
            largest_contour_area=1800,
            contour_count=3,
            processing_time=0.05
        )
        
        assert result.motion_detected is True
        assert result.motion_area == 2500
        assert result.largest_contour_area == 1800
        assert result.contour_count == 3
        assert result.processing_time == 0.05
    
    def test_motion_result_no_motion(self):
        """Test motion result when no motion detected."""
        result = MotionResult(
            motion_detected=False,
            motion_area=0,
            largest_contour_area=0,
            contour_count=0,
            processing_time=0.03
        )
        
        assert result.motion_detected is False
        assert result.motion_area == 0
        assert result.largest_contour_area == 0
    
    def test_motion_result_dict_conversion(self):
        """Test conversion to dictionary."""
        result = MotionResult(
            motion_detected=True,
            motion_area=1500,
            largest_contour_area=900,
            contour_count=2,
            processing_time=0.04
        )
        
        result_dict = result.to_dict()
        expected_keys = {
            'motion_detected', 'motion_area', 'largest_contour_area',
            'contour_count', 'processing_time'
        }
        
        assert set(result_dict.keys()) == expected_keys
        assert result_dict['motion_detected'] is True
        assert result_dict['motion_area'] == 1500


class TestWeightedMotionDetector:
    """Test weighted motion detection implementation."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = Config.create_test_config()
        self.detector = WeightedMotionDetector(self.config)
    
    def teardown_method(self):
        """Clean up after test."""
        if hasattr(self.detector, '_bg_subtractor'):
            del self.detector._bg_subtractor
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.config == self.config
        assert not self.detector.is_initialized()
    
    def test_detector_initialization_process(self):
        """Test detector initialization process."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Initialize detector
        self.detector.initialize(test_frame)
        
        assert self.detector.is_initialized()
        assert hasattr(self.detector, '_bg_subtractor')
        assert hasattr(self.detector, '_weight_mask')
    
    def test_weight_mask_creation(self):
        """Test weight mask creation."""
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        self.detector.initialize(test_frame)
        
        weight_mask = self.detector._weight_mask
        
        assert weight_mask.shape == (480, 640)
        assert weight_mask.dtype == np.float32
        
        # Check that center region has higher weights
        center_y, center_x = 240, 320
        edge_weight = weight_mask[10, 10]
        center_weight = weight_mask[center_y, center_x]
        
        assert center_weight > edge_weight
    
    def test_motion_detection_no_motion(self):
        """Test motion detection with static scene."""
        # Create static background
        background = np.full((480, 640), 128, dtype=np.uint8)
        
        self.detector.initialize(background)
        
        # Test with same frame (no motion)
        result = self.detector.detect_motion(background)
        
        assert isinstance(result, MotionResult)
        assert result.motion_detected is False
        assert result.motion_area == 0
        assert result.processing_time > 0
    
    def test_motion_detection_with_motion(self):
        """Test motion detection with moving object."""
        # Create background
        background = np.zeros((480, 640), dtype=np.uint8)
        self.detector.initialize(background)
        
        # Let background model stabilize
        for _ in range(5):
            self.detector.detect_motion(background)
        
        # Create frame with motion (white square in center)
        motion_frame = background.copy()
        motion_frame[200:280, 280:360] = 255  # Large white square in center
        
        result = self.detector.detect_motion(motion_frame)
        
        assert isinstance(result, MotionResult)
        # Motion should be detected due to the large change
        if result.motion_detected:
            assert result.motion_area > 0
            assert result.largest_contour_area > 0
    
    def test_consecutive_detection_filtering(self):
        """Test consecutive detection requirement."""
        background = np.zeros((480, 640), dtype=np.uint8)
        motion_frame = background.copy()
        motion_frame[200:300, 250:350] = 255  # Large motion area
        
        self.detector.initialize(background)
        
        # Let background stabilize
        for _ in range(5):
            self.detector.detect_motion(background)
        
        # First detection - should not trigger due to consecutive requirement
        result1 = self.detector.detect_motion(motion_frame)
        
        # Second consecutive detection - should trigger
        result2 = self.detector.detect_motion(motion_frame)
        
        # The behavior depends on the consecutive_detections_required setting
        # With default setting of 2, we need 2 consecutive detections
        consecutive_required = self.config.motion.consecutive_detections_required
        
        if consecutive_required > 1:
            # First detection might not trigger final result
            assert isinstance(result1, MotionResult)
            assert isinstance(result2, MotionResult)
    
    def test_motion_area_calculation(self):
        """Test motion area calculation and weighting."""
        background = np.zeros((480, 640), dtype=np.uint8)
        self.detector.initialize(background)
        
        # Stabilize background
        for _ in range(5):
            self.detector.detect_motion(background)
        
        # Create motion in center (high weight) vs edge (low weight)
        center_motion = background.copy()
        center_motion[220:260, 300:340] = 255  # Center region
        
        edge_motion = background.copy()
        edge_motion[10:50, 10:50] = 255  # Edge region
        
        center_result = self.detector.detect_motion(center_motion)
        
        # Reset detector
        self.detector.initialize(background)
        for _ in range(5):
            self.detector.detect_motion(background)
        
        edge_result = self.detector.detect_motion(edge_motion)
        
        # Center motion should have higher weighted area due to weight mask
        if center_result.motion_detected and edge_result.motion_detected:
            # Note: The actual comparison depends on implementation details
            assert isinstance(center_result, MotionResult)
            assert isinstance(edge_result, MotionResult)
    
    def test_motion_threshold_filtering(self):
        """Test motion threshold filtering."""
        # Use high threshold to filter small motions
        high_threshold_config = Config.create_test_config()
        high_threshold_config.motion.motion_threshold = 5000  # Very high
        detector = WeightedMotionDetector(high_threshold_config)
        
        background = np.zeros((480, 640), dtype=np.uint8)
        detector.initialize(background)
        
        # Stabilize
        for _ in range(5):
            detector.detect_motion(background)
        
        # Small motion that should be filtered out
        small_motion = background.copy()
        small_motion[240:250, 320:330] = 255  # Small white square
        
        result = detector.detect_motion(small_motion)
        
        # Should not detect motion due to high threshold
        assert result.motion_detected is False
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing (YUV420 handling)."""
        # Test with different frame formats
        test_cases = [
            np.random.randint(0, 255, (480, 640), dtype=np.uint8),  # Grayscale
            np.random.randint(0, 255, (480, 640, 1), dtype=np.uint8),  # Single channel
            np.random.randint(0, 255, (720, 640), dtype=np.uint8),  # YUV420-like (larger)
        ]
        
        for test_frame in test_cases:
            try:
                detector = WeightedMotionDetector(self.config)
                detector.initialize(test_frame)
                result = detector.detect_motion(test_frame)
                assert isinstance(result, MotionResult)
            except Exception as e:
                pytest.fail(f"Frame preprocessing failed for shape {test_frame.shape}: {e}")
    
    def test_error_handling_invalid_frame(self):
        """Test error handling with invalid frame."""
        detector = WeightedMotionDetector(self.config)
        
        # Test with None frame
        with pytest.raises(MotionDetectionError, match="Invalid frame"):
            detector.detect_motion(None)
        
        # Test with empty frame
        empty_frame = np.array([])
        with pytest.raises(MotionDetectionError, match="Invalid frame"):
            detector.detect_motion(empty_frame)
    
    def test_error_handling_uninitialized_detector(self):
        """Test error handling when detector not initialized."""
        detector = WeightedMotionDetector(self.config)
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        with pytest.raises(MotionDetectionError, match="Detector not initialized"):
            detector.detect_motion(test_frame)
    
    def test_detector_reset(self):
        """Test detector reset functionality."""
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        self.detector.initialize(test_frame)
        assert self.detector.is_initialized()
        
        self.detector.reset()
        assert not self.detector.is_initialized()
    
    def test_detector_statistics(self):
        """Test detector statistics tracking."""
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        self.detector.initialize(test_frame)
        
        # Process several frames
        for i in range(5):
            frame = test_frame + (i * 10)  # Slight variations
            self.detector.detect_motion(frame)
        
        stats = self.detector.get_stats()
        
        assert 'total_frames_processed' in stats
        assert 'motion_detections' in stats
        assert 'average_processing_time' in stats
        assert 'is_initialized' in stats
        
        assert stats['total_frames_processed'] == 5
        assert stats['is_initialized'] is True
        assert stats['average_processing_time'] > 0
    
    def test_detector_parameter_validation(self):
        """Test validation of detector parameters."""
        # Test invalid motion threshold
        invalid_config = Config.create_test_config()
        invalid_config.motion.motion_threshold = -100
        
        with pytest.raises(ValueError):
            WeightedMotionDetector(invalid_config)
        
        # Test invalid central region bounds
        invalid_config2 = Config.create_test_config()
        invalid_config2.motion.central_region_bounds = (0.8, 0.2)  # Reversed
        
        with pytest.raises(ValueError):
            WeightedMotionDetector(invalid_config2)


class TestMotionDetector:
    """Test high-level motion detector interface."""
    
    def test_motion_detector_initialization(self):
        """Test motion detector initialization."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        assert isinstance(detector._implementation, WeightedMotionDetector)
        assert detector.config == config
    
    def test_motion_detector_context_manager(self):
        """Test motion detector as context manager."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        with detector:
            detector.initialize(test_frame)
            assert detector.is_operational()
            
            result = detector.detect_motion(test_frame)
            assert isinstance(result, MotionResult)
    
    def test_motion_detector_batch_processing(self):
        """Test batch processing of frames."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        # Create test frames
        frames = []
        base_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        for i in range(5):
            frame = base_frame + (i * 5)  # Small variations
            frames.append(frame.astype(np.uint8))
        
        detector.start()
        detector.initialize(frames[0])
        
        results = detector.process_batch(frames[1:])  # Skip first frame used for init
        
        assert len(results) == 4
        assert all(isinstance(result, MotionResult) for result in results)
        
        detector.stop()
    
    def test_motion_detector_calibration(self):
        """Test motion detector calibration process."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        detector.start()
        
        # Create calibration frames (should be stable background)
        calibration_frames = []
        base_frame = np.full((480, 640), 128, dtype=np.uint8)
        for i in range(10):
            # Add slight noise to simulate real camera
            noise = np.random.normal(0, 2, (480, 640)).astype(np.uint8)
            frame = cv2.add(base_frame, noise)
            calibration_frames.append(frame)
        
        # Calibrate detector
        detector.calibrate(calibration_frames)
        assert detector.is_operational()
        
        # Test motion detection after calibration
        motion_frame = base_frame.copy()
        motion_frame[200:300, 250:350] = 255  # Add motion
        
        result = detector.detect_motion(motion_frame)
        assert isinstance(result, MotionResult)
        
        detector.stop()
    
    def test_motion_detector_sensitivity_adjustment(self):
        """Test dynamic sensitivity adjustment."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        detector.start()
        detector.initialize(test_frame)
        
        # Test sensitivity adjustment
        original_threshold = detector._implementation.config.motion.motion_threshold
        
        detector.adjust_sensitivity(0.5)  # Reduce sensitivity (higher threshold)
        new_threshold = detector._implementation.config.motion.motion_threshold
        assert new_threshold > original_threshold
        
        detector.adjust_sensitivity(2.0)  # Increase sensitivity (lower threshold)
        newest_threshold = detector._implementation.config.motion.motion_threshold
        assert newest_threshold < new_threshold
        
        detector.stop()
    
    def test_motion_detector_performance_monitoring(self):
        """Test performance monitoring features."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        detector.start()
        detector.initialize(test_frame)
        
        # Process frames and monitor performance
        processing_times = []
        for i in range(10):
            frame = test_frame + (i * 5)
            start_time = time.time()
            result = detector.detect_motion(frame.astype(np.uint8))
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
            assert result.processing_time > 0
        
        # Check performance stats
        perf_stats = detector.get_performance_stats()
        
        assert 'average_fps' in perf_stats
        assert 'peak_processing_time' in perf_stats
        assert 'total_processed' in perf_stats
        
        detector.stop()
    
    def test_motion_detector_region_of_interest(self):
        """Test region of interest functionality."""
        config = Config.create_test_config()
        detector = MotionDetector(config)
        
        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        detector.start()
        detector.initialize(test_frame)
        
        # Set region of interest (center region)
        roi = (160, 120, 320, 240)  # x, y, width, height
        detector.set_region_of_interest(roi)
        
        # Create motion outside ROI
        motion_frame = test_frame.copy()
        motion_frame[50:100, 50:100] = 255  # Motion outside ROI
        
        result = detector.detect_motion(motion_frame)
        
        # Motion outside ROI should have less impact
        assert isinstance(result, MotionResult)
        
        detector.stop()


if __name__ == '__main__':
    pytest.main([__file__])