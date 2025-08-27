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

### Text editor preference
Use VIM instead of nano for console edits.

## Architecture

The system follows a modular architecture with four main components:

- **`wildlife_camera.py`**: Main orchestrator that coordinates all components and manages the event loop
- **`config.py`**: Centralized configuration management using environment variables (.env file)
- **`camera_manager.py`**: Handles dual-stream camera operations (high-res capture + low-res motion detection)
- **`motion_detector.py`**: OpenCV-based motion detection with central region weighting and consecutive detection filtering

### Data Flow

1. **Motion Detection Loop**: Low-resolution frames (160x120) captured continuously for motion analysis
2. **Motion Processing**: Background subtraction → thresholding → contour analysis → central region filtering
3. **Photo Capture**: High-resolution frames (1920x1080) captured only when motion is detected
4. **Telegram Integration**: Async photo transmission with automatic cleanup

### Key Configuration Parameters

All configuration is centralized in `Config` class:

- **Motion Detection**: `motion_threshold` (30), `min_contour_area` (50), `consecutive_detections_required` (2)
- **Camera**: Dual resolution streams with frame rate limiting (`frame_duration`: 100000 microseconds)
- **Timing**: `cooldown_period` (10s), `frame_interval` (0.2s for 5 FPS)
- **Storage**: `max_images` (100) with automatic cleanup of oldest files

### Environment Setup

Requires `.env` file with:
- `TELEGRAM_BOT_TOKEN`: Bot token for Telegram integration
- `TELEGRAM_CHAT_ID`: Target chat/channel ID

### Hardware Dependencies

- Raspberry Pi 5 with camera module
- Uses Picamera2 for native Pi camera control
- OpenCV for computer vision processing

### Error Handling

The system includes comprehensive error handling:
- Main loop continues on individual frame errors
- Automatic camera cleanup on exit
- Graceful degradation for Telegram failures
- File system error protection during cleanup

## Development Notes

- The system uses async/await for Telegram operations while maintaining synchronous camera operations
- Motion detection uses weighted masks to prioritize central regions
- Consecutive detection filtering reduces false positives
- Background subtraction model (MOG2) automatically adapts to lighting changes