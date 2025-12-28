# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, identifies species using Google SpeciesNet AI, and sends notifications to a Telegram channel. The system uses OpenCV for motion detection, Picamera2 for camera control, and SpeciesNet for AI-powered species identification.

## Key Commands

### Camera preview for focus adjustment
```bash
python3 scripts/camera_preview.py
```
Starts MJPEG web stream on port 8000 for adjusting camera focus and positioning. Access at `http://<pi-ip>:8000`.

### Running the system
```bash
uv run python src/wildlife_system.py
```

### Package Management

This project uses **UV** for fast, reliable Python package management with Python 3.13.

**Note**: UV binary is located at `~/.local/bin/uv` and should already be in your PATH from `.bashrc`. If running scripts in non-interactive shells (e.g., cron jobs, systemd services), you may need to explicitly set PATH:
```bash
export PATH="$HOME/.local/bin:/usr/bin:$PATH"
```

**Installing/syncing dependencies**:
```bash
uv sync
```

**Running Python scripts with UV**:
```bash
uv run python scripts/test_classification.py
uv run python src/wildlife_system.py
```

**Running tests**:
```bash
uv run pytest tests/ -v
```

**Testing species classification**:
```bash
uv run python scripts/test_classification.py
```
This captures a photo and runs the full SpeciesNet pipeline (MegaDetector + species classifier). First run downloads ~214MB model files from Kaggle.

### Running specific test files
```bash
uv run pytest tests/test_config.py -v
uv run pytest tests/test_camera_manager.py -v
uv run pytest tests/test_motion_detector.py -v
```

### Text editor preference
Use VIM instead of nano for console edits.

## Architecture

The system follows a modular architecture with these main components:

- **`wildlife_system.py`**: Main orchestrator that coordinates all components and manages the event loop
- **`config.py`**: Centralized configuration management using environment variables (.env file)
- **`camera_manager.py`**: Handles dual-stream camera operations (high-res capture + low-res motion detection)
- **`motion_detector.py`**: OpenCV-based motion detection with central region weighting and consecutive detection filtering
- **`species_identifier.py`**: SpeciesNet AI integration for wildlife species identification
- **`database_manager.py`**: SQLite database for detection logging
- **`notification_service.py`**: Telegram notification service with message formatting
- **`resource_manager.py`**: Memory management, storage cleanup, and system monitoring
- **`models.py`**: Consolidated data models (MotionResult, DetectionResult, IdentificationResult, DetectionRecord)
- **`exceptions.py`**: Unified exception hierarchy for all components
- **`utils.py`**: Utilities (PerformanceTimer, MotionVisualizer, SharpnessAnalyzer, SunChecker)
- **`scripts/camera_preview.py`**: MJPEG streaming server for live camera preview (focus adjustment tool)

### Data Flow

1. **Motion Detection Loop**: Low-resolution frames (640x480) captured continuously for motion analysis using YUV420 format
2. **Motion Processing**: Background subtraction → thresholding → contour analysis → central region filtering
3. **Photo Capture**: High-resolution frames (1920x1080) captured only when motion is detected
4. **Species Identification**: SpeciesNet AI analyzes captured image (~5-10 seconds processing time)
5. **Database Logging**: Detection stored with species name, confidence score, and metadata
6. **Telegram Notification**: Async notification with species information
7. **Cleanup**: Automatic old image cleanup to manage storage

### Key Configuration Parameters

All configuration is centralized in `Config` class with nested dataclasses:

- **Motion Detection**: `motion_threshold` (2000), `min_contour_area` (50), `consecutive_detections_required` (2)
- **Camera**: Dual resolution streams with frame rate limiting (`frame_duration`: 100000 microseconds)
  - **Exposure Control**: `exposure_time` (2000μs = 1/500s) and `analogue_gain` (2.5x) for motion freeze
  - Set either to `None` to enable auto-exposure mode
- **Timing**: `cooldown_period` (30s), `frame_interval` (0.2s for 5 FPS)
- **Storage**: `max_images` (100) with automatic cleanup of oldest files
- **Species Identification**: `model_version` (v4.0.1a), `country_code` (DEU), `admin1_region` (NW), `unknown_species_threshold` (0.5)

### Configuration Architecture

The system uses a sophisticated configuration system with:
- **Type-safe dataclasses**: `CameraConfig`, `MotionConfig`, `PerformanceConfig`, `StorageConfig`, `SpeciesConfig`
- **Environment variable overrides**: All parameters can be overridden via env vars
- **Validation**: Configuration validation with meaningful error messages
- **Test configuration factory**: `Config.create_test_config()` for unit tests

### Species Identification Architecture

The species identification system integrates Google SpeciesNet v5.0.2:

- **SpeciesIdentifier**: Main class that wraps SpeciesNet (uses `SpeciesNet` class, not `SpeciesNetEnsemble`)
- **API**: Uses `predict()` method with `filepaths`, `country`, and `admin1_region` parameters
- **Model**: Default is `kaggle:google/speciesnet/pyTorch/v4.0.1a/1` (auto-downloaded from Kaggle on first use)
- **Components**: Loads detector (MegaDetector), classifier, and ensemble combiner
- **Lazy Loading**: Model loads on first identification request (not at startup) - takes ~6 seconds
- **Geographic Filtering**: Configured for Bonn, Germany (DEU/NW region) via geofencing
- **Confidence Thresholds**: Two-stage filtering (detection @ 0.6, classification @ 0.5)
- **Error Resilience**: Always returns valid IdentificationResult, never crashes
- **MockSpeciesIdentifier**: Test implementation for development without SpeciesNet

**SpeciesNet Dependencies**:
- Requires `ml-dtypes>=0.5.0` for float4_e2m1fn support
- Requires `numpy>=2.1.0` (ml-dtypes 0.5+ dependency)
- Requires `opencv-python>=4.10.0` (for NumPy 2.x compatibility)
- Uses ONNX for model inference (PyTorch backend)

### Camera Manager Architecture

The camera system supports multiple implementations through the `CameraInterface`:
- **PiCameraManager**: Production implementation using Picamera2 with error handling and resource management
- **MockCameraManager**: Test implementation for development without hardware
- **Dual-stream capture**: Separate low-res motion detection and high-res photo capture
- **Resource management**: Automatic cleanup and memory management for Pi Zero compatibility
- **Camera Preview Tool**: `scripts/camera_preview.py` provides MJPEG web streaming for focus adjustment (based on official Picamera2 example)

### Motion Detection Strategy

- **Background subtraction**: Uses MOG2 algorithm that adapts to lighting changes
- **Central region weighting**: Emphasizes motion in the center of the frame
- **Consecutive detection filtering**: Reduces false positives by requiring multiple detections
- **Contour analysis**: Validates motion based on size and position

### Environment Setup

Requires `.env` file with:
- `TELEGRAM_BOT_TOKEN`: Bot token for Telegram integration
- `TELEGRAM_CHAT_ID`: Target chat/channel ID

Additional optional environment variables for fine-tuning:
- Motion: `MOTION_THRESHOLD`, `MOTION_CONSECUTIVE_REQUIRED`, `MOTION_FRAME_INTERVAL`
- Camera: `CAMERA_MAIN_RESOLUTION`, `CAMERA_MOTION_RESOLUTION`, `CAMERA_EXPOSURE_TIME`, `CAMERA_ANALOGUE_GAIN`
- Performance: `PERFORMANCE_COOLDOWN`, `PERFORMANCE_MAX_IMAGES`
- Species: `SPECIES_COUNTRY_CODE`, `SPECIES_REGION`, `SPECIES_UNKNOWN_THRESHOLD`

### Hardware Dependencies

- **Raspberry Pi 5 with 8GB RAM** (required for SpeciesNet)
- **Raspberry Pi Camera Module** (any compatible module, tested with IMX477)
- **Storage**: ~2GB for models and images (SpeciesNet models: ~214MB)
- **Python 3.13** with system site packages access (for libcamera)
- Uses Picamera2 for native Pi camera control (requires libcamera system package)
- OpenCV 4.11+ for computer vision processing
- SpeciesNet 5.0.2 for AI species identification (PyTorch-based, ONNX runtime)

### Error Handling

The system includes comprehensive error handling:
- **Camera recovery**: Automatic restart on repeated errors
- **Resource cleanup**: Proper memory management with SystemMonitor
- **Graceful degradation**: System continues on individual component failures
- **Species ID fallback**: Returns "Unknown species" on any identification error
- **Logging**: Comprehensive logging for debugging all components

### Testing Strategy

The codebase includes extensive unit tests:
- **Configuration validation tests**: Ensure proper parameter validation including SpeciesConfig
- **Camera manager tests**: Test both production and mock implementations
- **Motion detection tests**: Validate detection algorithms
- **Species identification tests**: Mock SpeciesNet calls for unit testing
- **Database tests**: Validate detection logging and queries
- **Integration tests**: End-to-end system testing

Test files follow the pattern `test_*.py` and use pytest with asyncio support.
MockSpeciesIdentifier is available for testing without SpeciesNet dependency.

## Development Notes

- The system uses async/await for Telegram operations while maintaining synchronous camera and AI operations
- Motion detection uses weighted masks to prioritize central regions
- Background subtraction model automatically adapts to lighting changes
- Camera manager provides both production (Picamera2) and mock implementations for development
- **Species identification uses lazy loading** - model loads on first detection, not at startup
- SpeciesNet ensemble combines MegaDetector (object detection) with species classifier
- Geographic filtering automatically restricts predictions to region-appropriate species
- Configuration system supports both defaults and environment-based overrides
- All major components are thoroughly unit tested
- **Performance**: SpeciesNet inference takes ~17 seconds on Pi 5 CPU (no GPU acceleration)
  - Model loading: ~6 seconds (first time only, cached afterward)
  - Detection + classification: ~11 seconds per image
- **Memory**: System uses ~2-3GB RAM during species identification

### UV and Virtual Environment Setup

The project uses UV with a Python 3.13 virtual environment that has system site packages enabled (required for libcamera access):

```bash
# Virtual environment is at .venv with system-site-packages = true
# This allows access to system-installed libcamera Python bindings
# To recreate if needed:
uv venv --python /usr/bin/python3 --system-site-packages
uv sync
```

The `.venv/pyvenv.cfg` file should have `include-system-site-packages = true`.

### Common Dependency Issues

**ml-dtypes and NumPy compatibility**:
- ml-dtypes 0.5.0+ requires NumPy 2.1.0+
- Older OpenCV versions (< 4.10) don't support NumPy 2.x
- Solution: Use opencv-python >= 4.10.0 with numpy >= 2.1.0

**PATH issues in non-interactive shells**:
- UV is at `~/.local/bin/uv` (already in PATH for interactive shells)
- For cron jobs or systemd services, explicitly set: `PATH=$HOME/.local/bin:/usr/bin:$PATH`
- Note: Interactive terminal sessions via `.bashrc` have this configured correctly

**libcamera access**:
- libcamera is a system package (python3-libcamera) from apt
- UV venv must have system-site-packages enabled
- Python 3.13 is required (matches system Python version)