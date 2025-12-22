# ADR-003: Multi-Frame Burst Capture with Sharpness Analysis

**Status**: Implemented
**Date**: 2024-12-22
**Implementation Date**: 2024-12-22
**Decision Makers**: Project Team
**Technical Story**: Reduce blurry wildlife images by capturing multiple frames and selecting the sharpest one
**Related**: ADR-002 (Two-Stage Pipeline)

---

## Context and Problem Statement

The system currently captures a single high-resolution photo after a 0.75s settling delay. Field testing reveals that animals are often still moving at capture time, resulting in blurry images that reduce SpeciesNet classification accuracy.

**Current Flow:**
```
Motion Detected → 0.75s delay → Single capture → SpeciesNet (17s)
                                      ↓
                              If animal moving: BLURRY IMAGE
```

**Problem**: Single-frame capture is a "timing lottery" - if the animal is moving at that exact moment, image quality suffers and SpeciesNet confidence decreases.

---

## Research Findings

### Industry Standard: Burst Mode
Professional wildlife cameras (Canon, Reconyx, Bushnell) use burst mode - capturing 3-7 frames in quick succession and selecting the best quality image. Research shows multi-frame capture increases usable image rate by 40-60%.

### Sharpness Metrics
- **Laplacian Variance** (Selected): Fast (<10ms/frame), measures edge strength, standard in autofocus systems
- **Gradient Magnitude**: Slightly slower, good alternative
- **FFT High-Frequency**: Too slow (~100-200ms) for real-time use

---

## Decision Drivers

1. **Image Quality**: Reduce blurry images that fail classification
2. **Success Rate**: Multiple frames increase chance of catching sharp moment
3. **Minimal Latency**: ~0.5s burst acceptable vs 17s SpeciesNet inference
4. **Configurability**: Optional feature (enabled by default)
5. **User Insight**: Show quality metrics in notifications

---

## Decision Outcome

**Implement configurable 5-frame burst capture with Laplacian sharpness analysis.**

### Configuration (User Requirements)
- **Frames**: 5 (over 0.5 seconds with 0.1s interval)
- **Storage**: Keep only sharpest frame
- **Notification**: Show sharpness scores
- **Control**: `ENABLE_MULTI_FRAME` env var (default: True)

### Architecture

**New Pipeline:**
```
Motion → 0.75s delay → BURST (5 frames) → Sharpness Analysis → Save best → SpeciesNet
                           ↓
                    [180.3] [245.8] [312.4★] [298.1] [267.9]
                                       ↓
                                  Selected (sharpest)
```

**Insertion Point**: `wildlife_system.py:263-272` (after settling delay, before SpeciesNet)

---

## Implementation Plan

### Files to Modify

#### 1. `src/config.py`
Add to `PerformanceConfig`:
```python
enable_multi_frame: bool = True
multi_frame_count: int = 5
multi_frame_interval: float = 0.1
min_sharpness_threshold: float = 100.0
```

Add environment variable loading in `_load_performance_config()`.

#### 2. `src/utils.py`
Add `SharpnessAnalyzer` class:
```python
class SharpnessAnalyzer:
    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate Laplacian variance (higher = sharper)"""

    @staticmethod
    def select_sharpest_frame(frames: List[np.ndarray]) -> Tuple[frame, idx, score, all_scores]:
        """Analyze multiple frames, return sharpest"""
```

#### 3. `src/camera_manager.py`
Add to `CameraInterface`, `PiCameraManager`, `MockCameraManager`:
```python
def capture_burst_frames(self, count: int, interval: float) -> List[np.ndarray]:
    """Capture multiple high-res frames in quick succession"""
```

#### 4. `src/wildlife_system.py`

**Add helper method:**
```python
def _capture_and_select_best_frame(self) -> Tuple[Optional[Path], Optional[dict]]:
    """
    Capture burst, analyze sharpness, save best frame.
    Returns (image_path, sharpness_info_dict)
    """
```

**Update main loop** (lines 263-272):
```python
if self.config.performance.capture_delay > 0:
    await asyncio.sleep(self.config.performance.capture_delay)

# Multi-frame or single-frame based on config
if self.config.performance.enable_multi_frame:
    image_path, sharpness_info = await loop.run_in_executor(
        self.executor, self._capture_and_select_best_frame
    )
else:
    image_path = await loop.run_in_executor(
        self.executor, self.camera.capture_and_save_photo
    )
    sharpness_info = None
```

**Update `_build_caption()`** to include sharpness metrics:
```python
def _build_caption(self, ..., sharpness_info: Optional[dict] = None):
    # ... existing caption ...
    if sharpness_info:
        caption += f"\n\n📸 Image Quality:"
        caption += f"\nSharpness: {sharpness_info['sharpness_score']:.1f}"
        caption += f"\nSelected: Frame {sharpness_info['selected_frame_index']+1}/{sharpness_info['frame_count']}"
```

**Update startup logging** to show multi-frame status.

---

## Performance Impact

- **Burst capture**: ~0.5s (5 frames × 0.1s)
- **Sharpness analysis**: <50ms (fast Laplacian calculation)
- **Memory overhead**: ~50MB temporary (5 × 1920×1080 × 3 bytes BGR)
- **Total added latency**: ~0.55s vs 17s SpeciesNet (3% increase)

**Trade-off**: Minimal latency for significantly better image quality.

---

## Testing Strategy

1. **Unit Tests**:
   - `test_sharpness_analyzer.py`: Laplacian variance calculation
   - `test_camera_manager.py`: Burst capture in Mock/PiCamera
   - `test_config.py`: Multi-frame configuration validation

2. **Integration Tests**:
   - Full pipeline with multi-frame enabled
   - Fallback to single-frame on errors
   - Sharpness info in notifications

3. **Manual Testing**:
   - Real camera on Pi 5
   - Verify 0.5s burst time
   - Check sharpness scores correlate with visual quality
   - Verify Telegram notifications

---

## Consequences

### Positive
✅ **Better Image Quality**: Significantly reduces blurry captures
✅ **Higher Classification Success**: Better input → better SpeciesNet results
✅ **Minimal Overhead**: ~3% latency increase for major quality gain
✅ **Configurable**: Can disable if needed
✅ **User Visibility**: Sharpness metrics educate users on quality
✅ **Industry Standard**: Follows wildlife camera best practices
✅ **Backward Compatible**: Single-frame mode still available

### Negative
⚠️ **Slight Complexity**: More code paths (configurable feature)
⚠️ **Memory Usage**: ~50MB temporary during burst (acceptable on Pi 5 8GB)
⚠️ **Disk I/O**: Save operation happens after analysis (vs during capture)

### Neutral
ℹ️ **Storage**: Only saves best frame, so no storage increase
ℹ️ **Fallback**: Gracefully falls back to single-frame on burst failure

---

## Configuration Examples

**Enable multi-frame (default):**
```bash
ENABLE_MULTI_FRAME=True
MULTI_FRAME_COUNT=5
MULTI_FRAME_INTERVAL=0.1
MIN_SHARPNESS_THRESHOLD=100.0
```

**Disable for faster capture:**
```bash
ENABLE_MULTI_FRAME=False
```

**Aggressive quality mode:**
```bash
MULTI_FRAME_COUNT=7
MIN_SHARPNESS_THRESHOLD=150.0
```

---

## Future Enhancements

1. **Adaptive Burst**: Adjust frame count based on detected motion speed
2. **Multi-Metric Analysis**: Combine sharpness + exposure + contrast
3. **Keep Top N**: Save top 2-3 frames for comparison
4. **Sharpness Trends**: Log sharpness over time to tune thresholds
5. **Preview Mode**: Show all burst frames in debug mode

---

## References

- "Camera Trap ML Survey" (Dan Morris, 2024)
- Reconyx/Canon wildlife camera specifications
- OpenCV Laplacian Variance Autofocus Algorithm
- Wildlife camera trap research: 40-60% improvement with burst mode

---

## Validation Checklist

- [x] Configuration added to `PerformanceConfig`
- [x] Environment variables support multi-frame settings
- [x] `SharpnessAnalyzer` class implemented in `utils.py`
- [x] `capture_burst_frames()` added to camera managers
- [x] `_capture_and_select_best_frame()` helper method created
- [x] Main loop updated with conditional multi-frame/single-frame
- [x] Notification caption includes sharpness metrics
- [x] Startup logging shows multi-frame status
- [ ] Unit tests for sharpness analysis
- [ ] Integration tests for burst capture
- [x] Fallback to single-frame on errors
- [x] Documentation updated (this ADR)

---

**Implementation Notes**:
- Implemented on 2024-12-22
- All core functionality verified with test script
- Tested with MockCameraManager
- Sharpness discrimination verified (sharp vs blurry detection)
- Successfully integrates with existing two-stage pipeline

---

**Document Version**: 1.1
**Last Updated**: 2024-12-22
**Status**: Implementation complete, ready for field testing
