# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, and sends them to a Telegram channel. The system uses OpenCV for motion detection and the Picamera2 library for camera control.

## Key Commands

### Running the system
```bash
python src/wildlife_camera.py
```

### Installing dependencies
```bash
pip install -r requirements.txt
```

### Running tests
```bash
python -m pytest tests/ -v
```

### Running specific test files
```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_camera_manager.py -v
python -m pytest tests/test_motion_detector.py -v
```

### Text editor preference
Use VIM instead of nano for console edits.

## Architecture

The system follows a modular architecture with four main components:

- **`wildlife_camera.py`**: Main orchestrator that coordinates all components and manages the event loop
- **`config.py`**: Centralized configuration management using environment variables (.env file)
- **`camera_manager.py`**: Handles dual-stream camera operations (high-res capture + low-res motion detection)
- **`motion_detector.py`**: OpenCV-based motion detection with central region weighting and consecutive detection filtering

### Data Flow

1. **Motion Detection Loop**: Low-resolution frames (640x480) captured continuously for motion analysis using YUV420 format
2. **Motion Processing**: Background subtraction → thresholding → contour analysis → central region filtering
3. **Photo Capture**: High-resolution frames (1920x1080) captured only when motion is detected
4. **Telegram Integration**: Async photo transmission with automatic cleanup

### Key Configuration Parameters

All configuration is centralized in `Config` class with nested dataclasses:

- **Motion Detection**: `motion_threshold` (2000), `min_contour_area` (50), `consecutive_detections_required` (2)
- **Camera**: Dual resolution streams with frame rate limiting (`frame_duration`: 100000 microseconds)
- **Timing**: `cooldown_period` (30s), `frame_interval` (0.2s for 5 FPS)
- **Storage**: `max_images` (100) with automatic cleanup of oldest files

### Configuration Architecture

The system uses a sophisticated configuration system with:
- **Type-safe dataclasses**: `CameraConfig`, `MotionConfig`, `PerformanceConfig`, `StorageConfig`
- **Environment variable overrides**: All parameters can be overridden via env vars
- **Validation**: Configuration validation with meaningful error messages
- **Test configuration factory**: `Config.create_test_config()` for unit tests

### Camera Manager Architecture

The camera system supports multiple implementations through the `CameraInterface`:
- **PiCameraManager**: Production implementation using Picamera2 with error handling and resource management
- **MockCameraManager**: Test implementation for development without hardware
- **Dual-stream capture**: Separate low-res motion detection and high-res photo capture
- **Resource management**: Automatic cleanup and memory management for Pi Zero compatibility

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
- `MOTION_THRESHOLD`, `CAMERA_MAIN_RESOLUTION`, `PERFORMANCE_COOLDOWN`, etc.

### Hardware Dependencies

- Raspberry Pi 5 with camera module
- Uses Picamera2 for native Pi camera control
- OpenCV for computer vision processing

### Error Handling

The system includes comprehensive error handling:
- **Camera recovery**: Automatic restart on repeated errors
- **Resource cleanup**: Proper memory management for Pi Zero
- **Graceful degradation**: System continues on individual component failures
- **Logging**: Comprehensive logging for debugging

### Testing Strategy

The codebase includes extensive unit tests:
- **Configuration validation tests**: Ensure proper parameter validation
- **Camera manager tests**: Test both production and mock implementations
- **Motion detection tests**: Validate detection algorithms
- **Integration tests**: End-to-end system testing

Test files follow the pattern `test_*.py` and use pytest with asyncio support.

## Development Notes

- The system uses async/await for Telegram operations while maintaining synchronous camera operations
- Motion detection uses weighted masks to prioritize central regions
- Background subtraction model automatically adapts to lighting changes
- Camera manager provides both production (Picamera2) and mock implementations for development
- Configuration system supports both defaults and environment-based overrides
- All major components are thoroughly unit tested