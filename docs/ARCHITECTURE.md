# Wildlife Camera System Architecture

## Overview
Raspberry Pi 5-based wildlife camera system with motion detection, AI-powered species identification via Google SpeciesNet, and Telegram notifications.

## Hardware Configuration
- **Device**: Raspberry Pi 5 (8GB RAM)
- **Camera**: Raspberry Pi Camera Module (any compatible module, tested with IMX477)
- **Setup**: Outdoor wildlife camera for garden monitoring
- **Network**: Wi-Fi connection for Telegram notifications
- **Processing**: All AI inference runs locally (no cloud dependencies)

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│  Motion         │    │  Camera         │    │  Species            │
│  Detection      │    │  Management     │    │  Identification     │
│  (MOG2)         │    │  (Dual-stream)  │    │  (SpeciesNet AI)    │
│                 │    │                 │    │  - MegaDetector     │
│                 │    │                 │    │  - Classifier       │
└─────────────────┘    └─────────────────┘    └─────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐     │     ┌─────────────────┐
         │   Database      │     │     │   Notification  │
         │   (SQLite)      │     │     │   Service       │
         └─────────────────┘     │     └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Main Control   │
                    │  Loop           │
                    │  (async/await)  │
                    └─────────────────┘
```

### Data Flow

1. **Motion Detection**: Continuous monitoring at 640x480 (configurable FPS)
2. **Motion Trigger**: Configurable pixel threshold + consecutive detection filter
3. **Settling Delay**: 0.75s wait for camera stabilization
4. **Burst Capture**: Capture 5 high-res frames (1920x1080) in 0.5s
5. **Frame Selection**: Analyze sharpness + foreground content, select best frame
6. **AI Detection**: MegaDetector validates animals present (~2s)
7. **AI Classification**: SpeciesNet identifies species (~8-11s, only if animals detected)
8. **Data Logging**: Store detection with species, confidence, metadata in SQLite
9. **Telegram Notification**: Send species name, confidence, image quality metrics
10. **Cooldown**: 30-second pause before next detection cycle

## Technical Specifications

### Resolution Strategy
- **Motion Detection Stream**: 640x480 (low-res, continuous)
- **Capture Stream**: 1920x1080 (high-res, triggered)
- **Dual-Stream**: Picamera2 manages both streams simultaneously
- **Storage Format**: JPEG at full resolution (1920x1080)

### AI Integration (SpeciesNet v5.0.2)
- **Framework**: Google SpeciesNet (local PyTorch/ONNX inference)
- **Model**: v4.0.1a (always-crop variant) with geographic filtering
- **Geographic Filter**: Germany (DEU) / North Rhine-Westphalia (NW)
- **Two-Stage Pipeline**:
  - Stage 1: MegaDetector (animal detection with bounding boxes)
  - Stage 2: Species Classifier (species identification from crops)
- **Processing Time**: ~17s total (~6s model load first time, ~11s inference)
- **Confidence Thresholds**: Detection @ 0.6, Classification @ 0.5
- **Model Cache**: ~/.cache/speciesnet (~214MB)

### Performance Profile (Pi 5 8GB)
- **RAM Usage**: ~2-3GB during species identification
- **Processing Time**: ~17s per detection (MegaDetector + Classifier)
- **Early Exit Optimization**: Skip classifier if no animals detected (~8-10s saved)
- **Burst Capture Overhead**: ~0.55s (5 frames + sharpness analysis)
- **Memory Management**: Automatic cleanup, system monitoring

### Database Schema

```sql
-- Detection logs
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    motion_area INTEGER,
    species_name TEXT,
    confidence_score REAL,
    processing_time REAL,
    api_success BOOLEAN
);
```

## Component Details

### 1. Motion Detection (`motion_detector.py`)
- **Algorithm**: OpenCV MOG2 Background Subtractor
- **Features**: Adapts to lighting changes, shadow detection disabled
- **Filtering**:
  - Central region weighting (prioritizes center of frame)
  - Consecutive detection requirement (reduces false positives)
  - Optional color variance filtering (for uniform leaf motion)
- **Configurable Thresholds**: Motion area, contour size, detection count

### 2. Camera Management (`camera_manager.py`)
- **Implementation**: Picamera2 for native Pi camera control
- **Dual-Stream Configuration**:
  - Main stream: 1920x1080 for capture
  - Low-res stream: 640x480 for motion detection
- **Exposure Control**: Configurable exposure time and analogue gain (or auto-exposure)
- **Burst Capture**: Multi-frame capture with configurable interval
- **Mock Implementation**: Available for testing without hardware

### 3. Species Identification (`species_identifier.py`)
- **AI Framework**: Google SpeciesNet v5.0.2 (local inference)
- **Two-Stage Pipeline**:
  - `detect_animals()`: MegaDetector finds animals with bounding boxes
  - `classify_species()`: SpeciesNet classifier identifies species (only if animals detected)
- **Geographic Filtering**: Built-in filtering for Germany/NW region
- **Lazy Loading**: Model loads on first detection request (~6s)
- **Error Resilience**: Always returns valid result, never crashes
- **Mock Implementation**: Available for testing without SpeciesNet

### 4. Burst Capture & Frame Selection (`utils.py`)
- **Multi-Frame Capture**: 5 frames at 0.1s intervals (configurable)
- **Sharpness Analysis**: Laplacian variance measurement
- **Motion-Aware Selection**:
  - Analyzes foreground content (edge density, intensity variance, contour count)
  - Prefers frames with actual animals over sharp empty backgrounds
  - Falls back to sharpest frame if no foreground detected
- **Performance**: <50ms analysis time for 5-frame burst

### 5. Database Management (`database_manager.py`)
- **Engine**: SQLite with absolute path resolution
- **Operations**: Log detections with species, confidence, processing time
- **Thread-Safe**: Connection pooling for async operations
- **Schema**: Single `detections` table with all metadata

### 6. Notification Service (`notification_service.py`)
- **Async Integration**: python-telegram-bot with async/await
- **Features**:
  - Detection notifications with species, confidence, motion metrics
  - Photo upload with captions
  - Media groups (original + annotated motion visualization)
  - System status messages
- **Error Handling**: Graceful failure, continues logging even if Telegram fails

### 7. Resource Management (`resource_manager.py`)
- **MemoryManager**: RAM monitoring, garbage collection, threshold checking
- **StorageManager**: Image storage, cleanup, disk space monitoring
- **SystemMonitor**: Unified system status (CPU temp, memory, storage)
- **Automatic Cleanup**: Old images deleted when limit reached

### 8. Data Models (`models.py`)
- **MotionResult**: Motion detection results with area, confidence, contours
- **DetectionResult**: MegaDetector animal detection results
- **IdentificationResult**: Species classification results
- **DetectionRecord**: Database record structure

### 9. Exception Hierarchy (`exceptions.py`)
- **WildlifeSystemError**: Base exception for all system errors
- **HardwareError**: Camera and database errors
- **ProcessingError**: Motion detection and species ID errors
- **NotificationError**: Telegram communication errors

### 10. Configuration (`config.py`)
- **Type-Safe Dataclasses**: Nested config structure
  - `CameraConfig`: Resolution, exposure, frame rate
  - `MotionConfig`: Thresholds, filtering, consecutive detection
  - `PerformanceConfig`: Cooldown, burst capture, frame selection
  - `StorageConfig`: Paths, image limits, cleanup
  - `SpeciesConfig`: Model version, geographic filter, thresholds
- **Environment Overrides**: All settings configurable via .env
- **Validation**: Built-in validation with meaningful error messages
- **Test Factory**: `Config.create_test_config()` for unit tests

## Key Features

### Multi-Frame Burst Capture (ADR-003)
- Captures 5 high-resolution frames in 0.5 seconds
- Analyzes each frame for sharpness (Laplacian variance)
- Selects best frame before running species identification
- Significantly reduces blurry images from moving animals

### Motion-Aware Frame Selection
- Analyzes foreground content in addition to sharpness
- Prevents selecting sharp but empty frames
- Multi-metric analysis (edge density, intensity variance, contour count)
- Ensures selected frame actually contains the detected animal

### Two-Stage AI Pipeline (ADR-002)
- Stage 1: MegaDetector validates animals are present
- Stage 2: SpeciesNet classifier identifies species (only if animals detected)
- Early exit optimization saves ~8-10s when no animals detected
- Better performance and clearer pipeline logic

### Geographic Filtering
- SpeciesNet configured for Germany (DEU) / North Rhine-Westphalia (NW)
- Automatically filters predictions to regionally-appropriate species
- Reduces false positives from exotic species

## Performance Profile

### Typical Operation (Pi 5)
- **Detection Frequency**: Variable (depends on wildlife activity)
- **Processing Time**:
  - Motion detection: Continuous at ~5 FPS
  - Burst capture + selection: ~0.55s
  - MegaDetector: ~2s
  - Species classification: ~8-11s (only if animals detected)
  - Total: ~17-20s per detection with animal, ~3s without
- **Memory Usage**:
  - Baseline: ~500MB
  - During SpeciesNet inference: ~2-3GB
  - Peak: ~3GB (comfortable on 8GB Pi 5)
- **Storage**: ~1-2MB per detection (JPEG + database entry)

### Accuracy Expectations
- **MegaDetector**: 99.4% animal detection accuracy
- **SpeciesNet**: 94.5% species-level accuracy (trained on camera trap data)
- **Geographic Filtering**: Limits predictions to German wildlife
- **Confidence Thresholds**: Detection @ 0.6, Classification @ 0.5

## Error Handling & Recovery

### AI Model Failures
- **Lazy Loading**: Model loads on first use, startup unaffected by model issues
- **Graceful Degradation**: Always returns "Unknown species" on failures
- **Error Logging**: Full error details logged for debugging
- **No Crashes**: System continues operating even if SpeciesNet fails

### Camera Issues
- **Automatic Recovery**: Camera restart on repeated errors
- **Resource Cleanup**: Proper frame disposal to prevent memory leaks
- **Mock Fallback**: MockCameraManager available for testing without hardware

### Memory Management (Pi 5)
- **System Monitor**: Tracks RAM and disk usage
- **Automatic Cleanup**: Old images deleted when limit reached
- **Frame Cleanup**: Burst frames released immediately after selection
- **Sufficient Headroom**: 8GB RAM provides comfortable margin for 3GB peak usage

## Monitoring & Analytics

### Real-time Monitoring
- **System Resources**: RAM, disk usage via SystemMonitor
- **Detection Metrics**: Motion area, consecutive detections, trigger frequency
- **AI Performance**: Processing time, confidence scores, detection vs classification time
- **Image Quality**: Sharpness scores, foreground content, selected frame index

### Database Analytics
- **Detection History**: All detections logged with full metadata
- **Species Tracking**: Which species detected, when, and how often
- **Performance Trends**: Processing times, success rates over time
- **Quality Metrics**: Sharpness scores, confidence distributions

### Telegram Integration
- Real-time notifications with species, confidence, and quality metrics
- Media groups showing original + motion-annotated images
- System status messages on demand

## Testing Infrastructure

### Unit Tests
- Component-level tests for all major modules
- Mock implementations (MockCameraManager, MockSpeciesIdentifier)
- Configuration validation tests
- Database operations testing

### Integration Tests
- End-to-end pipeline testing
- Two-stage AI pipeline validation
- Burst capture and frame selection
- Telegram notification delivery

### Test Scripts
- [camera_preview.py](../scripts/camera_preview.py): Live camera preview for focus adjustment
- [test_classification.py](../scripts/test_classification.py): Full SpeciesNet pipeline test
- [test_telegram.py](../scripts/test_telegram.py): Telegram bot integration test

## Documentation

### Architecture Decision Records (ADRs)
- **ADR-002**: Two-Stage Detection and Classification Pipeline
- **ADR-003**: Multi-Frame Burst Capture with Sharpness Analysis

### Configuration
- All settings documented in [CLAUDE.md](../CLAUDE.md)
- Environment variable overrides for production deployment
- Test configuration factory for unit tests

---

**Last Updated**: 2024-12-24
**System Version**: Pi 5 with SpeciesNet v5.0.2
**Status**: Production-ready with comprehensive testing