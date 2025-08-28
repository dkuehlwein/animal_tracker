# Wildlife Camera System Architecture

## Overview
Raspberry Pi Zero 2 W-based garden wildlife detection system with motion detection, species identification via iNaturalist API, and Telegram notifications.

## Hardware Configuration
- **Device**: Raspberry Pi Zero 2 W (512MB RAM)
- **Camera**: 12.3MP Pi HQ Camera (4056x3040 max resolution)
- **Setup**: Indoor camera pointing at garden through window
- **Network**: Wi-Fi connection for API calls and Telegram

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Motion         │    │  Camera         │    │  Species        │
│  Detection      │    │  Management     │    │  Identification │
│  (MOG2)         │    │  (Dual-res)     │    │  (iNaturalist)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐     │     ┌─────────────────┐
         │   Database      │     │     │   Telegram      │
         │   (SQLite)      │     │     │   Notifications │
         └─────────────────┘     │     └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Main Control   │
                    │  Loop           │
                    └─────────────────┘
```

### Data Flow

1. **Motion Detection**: Continuous monitoring at 640x480 @ 10fps
2. **Motion Trigger**: 2000+ pixel area threshold activates capture
3. **High-res Capture**: Switch to 1920x1080 for species identification
4. **API Processing**: Send image to iNaturalist for species identification
5. **Telegram Alert**: Send species name and confidence as text message
6. **Data Logging**: Store results and metadata in SQLite database
7. **Cooldown**: 30-second pause before next detection cycle

## Technical Specifications

### Resolution Strategy
- **Motion Detection Stream**: 640x480 @ 10fps (continuous)
- **Species ID Capture**: 1920x1080 (triggered)
- **Image Optimization**: Max 2048px longest side, 90% JPEG quality for API
- **Storage Format**: Original resolution stored locally

### API Integration
- **Service**: iNaturalist Computer Vision API
- **Endpoint**: `https://api.inaturalist.org/v1/computervision/score_image`
- **Rate Limit**: 100 requests/minute (no authentication required)
- **Input**: Base64 encoded JPEG
- **Timeout**: 30 seconds per request
- **Retry Logic**: 3 attempts with exponential backoff

### Performance Constraints (Pi Zero 2 W)
- **RAM Limit**: 512MB (vs 4-8GB on Pi 4/5)
- **Processing Time**: 1-3 seconds per detection
- **Memory Management**: Process images in chunks, immediate cleanup
- **Fallback Strategy**: Reduce resolution if memory pressure detected

### Database Schema

```sql
-- Detection logs
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    motion_area INTEGER,
    api_success BOOLEAN,
    species_predictions TEXT,  -- JSON array
    confidence_score REAL,
    processing_time REAL
);

-- Species tracking
CREATE TABLE species (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scientific_name TEXT UNIQUE,
    common_name TEXT,
    first_detected DATETIME,
    detection_count INTEGER DEFAULT 1
);
```

## Component Details

### 1. Motion Detection (`motion_detector.py`)
- **Algorithm**: OpenCV MOG2 Background Subtractor
- **Features**: Adapts to lighting changes, shadow detection disabled
- **Filtering**: Central region weighting, consecutive detection requirement
- **Threshold**: 2000+ pixels minimum area

### 2. Camera Management (`camera_manager.py`)
- **Dual Stream**: Low-res continuous + high-res triggered
- **Format**: RGB888 for both streams
- **Frame Rate**: 10fps limit via FrameDurationLimits
- **Auto-switching**: Seamless transition between resolutions

### 3. Species Identification (`species_identifier.py`) - NEW
- **API Client**: HTTP requests with error handling
- **Image Processing**: Resize and optimize before upload
- **Response Parsing**: Extract top predictions with confidence scores
- **Fallback**: Graceful degradation on API failures

### 4. Database Management (`database_manager.py`) - NEW
- **Engine**: SQLite (no external dependencies)
- **Operations**: Insert detections, update species counts
- **Analytics**: Query patterns, species frequency
- **Maintenance**: Automatic cleanup of old records

### 5. Telegram Integration (Enhanced)
- **Change**: Send text messages instead of photos
- **Format**: Species name, confidence, timestamp
- **Speed**: Instant delivery vs slow photo uploads
- **Fallback**: Send "Unknown species" if API fails

### 6. Configuration (`config.py`) - Updated
```python
# New Pi Zero optimized settings
motion_detection_resolution = (640, 480)
api_capture_resolution = (1920, 1080)
motion_threshold = 2000  # pixels
cooldown_period = 30  # seconds
api_timeout = 30  # seconds
max_memory_usage = 80  # percent
```

## Implementation Phases

### Phase 1: Foundation
- Update configuration for new resolution requirements
- Enhance camera manager for optimized dual-resolution capture
- Create database management and image optimization modules

### Phase 2: API Integration  
- Implement iNaturalist species identification service
- Modify main loop to integrate API calls between capture and notification
- Update Telegram messaging to send species text instead of photos

### Phase 3: Optimization
- Add Pi Zero memory management and monitoring
- Implement graceful degradation for system stability
- Add performance analytics and health monitoring

## Performance Expectations

### Typical Operation
- **Detection Frequency**: 1-5 per hour (depends on garden activity)
- **Processing Time**: 2-4 seconds per detection (motion → notification)
- **Memory Usage**: 200-400MB during normal operation
- **Storage**: ~1MB per detection (image + metadata)

### Expected Limitations
- **Species Accuracy**: 70-85% for common UK garden animals
- **Image Quality**: Limited by Pi Zero processing power
- **API Dependency**: System degrades gracefully if iNaturalist unavailable
- **Memory Constraints**: May skip detections if memory >80% usage

## Error Handling & Recovery

### API Failures
- Retry logic with exponential backoff
- Fallback to "Unknown species detected" messages
- Continue local storage and database logging

### Camera Issues
- Automatic camera restart on capture failures
- Graceful resolution fallback (1920x1080 → 1280x720 → 640x480)
- System health monitoring

### Memory Management
- Garbage collection after large image operations
- Monitor RAM usage, skip processing if >80%
- Emergency cleanup routines

## Monitoring & Analytics

### Real-time Monitoring
- System resource usage (RAM, CPU, disk)
- API response times and success rates
- Detection frequency and patterns

### Historical Analysis
- Species identification over time
- Most active periods (time of day/season)
- API accuracy trends
- System performance metrics

This architecture balances the ambitious species identification goals with the practical constraints of Pi Zero 2 W hardware while maintaining system reliability and user experience.