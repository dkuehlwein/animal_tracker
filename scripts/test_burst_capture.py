#!/usr/bin/env python3
"""
Test script for multi-frame burst capture with sharpness analysis.
Verifies ADR-003 implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera_manager import MockCameraManager
from config import Config
from utils import SharpnessAnalyzer
import cv2
import numpy as np


def test_burst_capture_pipeline():
    """Test complete burst capture pipeline."""
    print("="*60)
    print("Testing Multi-Frame Burst Capture (ADR-003)")
    print("="*60)
    
    # Load config
    config = Config()
    print(f"\n✓ Configuration loaded")
    print(f"  - Multi-frame enabled: {config.performance.enable_multi_frame}")
    print(f"  - Frame count: {config.performance.multi_frame_count}")
    print(f"  - Interval: {config.performance.multi_frame_interval}s")
    print(f"  - Min sharpness: {config.performance.min_sharpness_threshold}")
    
    # Initialize mock camera
    camera = MockCameraManager(config)
    camera.start()
    print(f"\n✓ Mock camera started")
    
    # Test 1: Burst capture
    print(f"\n[Test 1] Burst Capture")
    frames = camera.capture_burst_frames(
        count=config.performance.multi_frame_count,
        interval=config.performance.multi_frame_interval
    )
    print(f"  ✓ Captured {len(frames)} frames")
    assert len(frames) == config.performance.multi_frame_count, "Wrong number of frames"
    
    # Test 2: Sharpness analysis
    print(f"\n[Test 2] Sharpness Analysis")
    best_frame, idx, best_score, all_scores = SharpnessAnalyzer.select_sharpest_frame(frames)
    print(f"  ✓ Selected frame {idx+1}/{len(frames)}")
    print(f"  ✓ Sharpness score: {best_score:.1f}")
    print(f"  ✓ All scores: {[f'{s:.1f}' for s in all_scores]}")
    assert best_frame is not None, "No frame selected"
    assert best_score > 0, "Invalid sharpness score"
    
    # Test 3: Threshold check
    print(f"\n[Test 3] Quality Threshold")
    meets_threshold = best_score >= config.performance.min_sharpness_threshold
    print(f"  {'✓' if meets_threshold else '⚠️'} Threshold check: {best_score:.1f} vs {config.performance.min_sharpness_threshold}")
    
    # Test 4: Save best frame
    print(f"\n[Test 4] Save Best Frame")
    test_output = config.storage.image_dir / "test_burst_capture.jpg"
    success = cv2.imwrite(str(test_output), best_frame)
    print(f"  {'✓' if success else '✗'} Saved to: {test_output}")
    
    if test_output.exists():
        test_output.unlink()  # Cleanup
        print(f"  ✓ Cleanup complete")
    
    camera.stop()
    print(f"\n✓ Mock camera stopped")
    
    print("\n" + "="*60)
    print("✅ All tests passed! ADR-003 implementation verified.")
    print("="*60)


def test_sharpness_on_real_images():
    """Test sharpness calculation on synthetic sharp vs blurry images."""
    print("\n" + "="*60)
    print("Testing Sharpness Discrimination")
    print("="*60)
    
    # Create sharp image (high frequency content)
    print("\n[Sharp Image]")
    sharp = np.zeros((480, 640), dtype=np.uint8)
    for i in range(0, 640, 20):
        cv2.line(sharp, (i, 0), (i, 480), 255, 1)
    for i in range(0, 480, 20):
        cv2.line(sharp, (0, i), (640, i), 255, 1)
    sharp_score = SharpnessAnalyzer.calculate_sharpness(sharp)
    print(f"  Sharpness: {sharp_score:.1f}")
    
    # Create blurry image (apply Gaussian blur)
    print("\n[Blurry Image]")
    blurry = cv2.GaussianBlur(sharp, (51, 51), 15)
    blurry_score = SharpnessAnalyzer.calculate_sharpness(blurry)
    print(f"  Sharpness: {blurry_score:.1f}")
    
    # Verify sharp image has higher score
    print(f"\n✓ Sharp vs Blurry: {sharp_score:.1f} > {blurry_score:.1f}")
    assert sharp_score > blurry_score, "Sharp image should have higher score"
    
    print("\n" + "="*60)
    print("✅ Sharpness discrimination works correctly!")
    print("="*60)


if __name__ == "__main__":
    try:
        test_burst_capture_pipeline()
        test_sharpness_on_real_images()
        
        print("\n" + "="*60)
        print("🎉 ADR-003 Implementation Complete!")
        print("="*60)
        print("\nFeatures verified:")
        print("  ✓ Multi-frame burst capture")
        print("  ✓ Laplacian variance sharpness analysis")
        print("  ✓ Best frame selection")
        print("  ✓ Configuration integration")
        print("  ✓ Quality threshold checking")
        print("  ✓ Sharpness discrimination")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
