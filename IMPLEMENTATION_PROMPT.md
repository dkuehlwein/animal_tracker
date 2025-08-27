# Wildlife Camera System Implementation Prompt

**For Claude Code**: Please update and rewrite our wildlife camera system based on the research and requirements below.

## Project Context

We have an existing Raspberry Pi 5 wildlife camera system that needs to be updated for:
- **Hardware**: Raspberry Pi Zero 2 W (512MB RAM)  
- **Camera**: 12.3MP Pi HQ Camera (4056x3040 max resolution)
- **Setup**: Indoor camera pointing at garden through window
- **Goal**: Motion detection → species identification → Telegram notifications → database logging

## Current System Analysis

**Existing codebase** (8 months old, Pi 5-based):
- `wildlife_camera.py`: Main orchestrator with Telegram photo sending
- `camera_manager.py`: Dual-stream camera management  
- `motion_detector.py`: MOG2 motion detection (good, keep this)
- `config.py`: Environment-based configuration

**What works**: Motion detection, camera management, Telegram integration  
**What needs updating**: Resolution settings, species identification integration, database logging

## New Requirements

### Core Functionality Changes
1. **Keep Telegram** but send species text messages, NOT photos
2. **Add species identification** with mock/placeholder for now ("Unknown species detected")
3. **Add SQLite database** logging for all detections  
4. **Update camera settings** for Pi Zero 2 W performance

### Technical Specifications

**Camera Configuration**:
- Motion detection stream: 640x480 YUV420 @ 10fps (not RGB888!)
- Species ID capture: 1920x1080 RGB888 (triggered only)
- Note: lores stream MUST be YUV420 format (tested and confirmed)

**Performance Optimizations for Pi Zero 2 W**:
- 512MB RAM constraint (vs 4GB+ on Pi 5)
- Motion threshold: 2000+ pixels (not 30)
- Cooldown period: 30 seconds between detections (not 10)
- Memory management: immediate cleanup after processing

**Database Schema** (SQLite):
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    motion_area INTEGER,
    species_name TEXT DEFAULT 'Unknown species',
    confidence_score REAL DEFAULT 0.0,
    processing_time REAL,
    api_success BOOLEAN DEFAULT FALSE
);

CREATE TABLE species (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    first_detected DATETIME,
    detection_count INTEGER DEFAULT 1
);
```

### New Data Flow
```
Motion Detection (640x480 YUV420) 
    ↓
High-res Capture (1920x1080 RGB888)
    ↓
Species Identification (MOCK: return "Unknown species", confidence=0.0)
    ↓
Telegram Text: "🔍 Unknown species detected at 15:42 (garden camera)"
    ↓
SQLite Database Log (with image path, motion area, timestamp)
    ↓
Cleanup (memory management, old images if >100 stored)
```

### Updated Configuration Parameters

```python
# Pi Zero 2 W optimized settings
motion_detection_resolution = (640, 480)  # YUV420 format required
motion_detection_format = "YUV420"       # Critical for lores stream
api_capture_resolution = (1920, 1080)    # RGB888 format
motion_threshold = 2000                  # pixels (much higher than Pi 5)
cooldown_period = 30                     # seconds between API calls
consecutive_detections_required = 2      # reduce false positives
max_images = 100                         # storage limit
database_path = "data/detections.db"     # SQLite database
```

## Implementation Tasks

### Phase 1: Update Existing Components

1. **Update `config.py`**:
   - Add new Pi Zero 2 W optimized parameters
   - Add database configuration
   - Update resolution and timing settings

2. **Enhance `camera_manager.py`**:
   - Fix YUV420 format for motion detection stream
   - Optimize for Pi Zero memory constraints
   - Add error handling for memory pressure

3. **Keep `motion_detector.py` mostly unchanged**:
   - Update resolution to 640x480  
   - Adjust threshold to 2000+ pixels
   - Handle YUV420 format input (extract Y channel)

### Phase 2: Add New Components

4. **Create `database_manager.py`**:
   - SQLite operations (create tables, insert detections, query species)
   - Database initialization and maintenance
   - Analytics queries for species tracking

5. **Create `species_identifier.py`**:
   - Mock implementation returning "Unknown species"
   - Interface designed for future API integration
   - Error handling and timeout simulation

6. **Update `wildlife_camera.py`**:
   - Integrate database logging
   - Change Telegram to send text instead of photos
   - Add memory management for Pi Zero
   - Integrate species identification in main loop

### Phase 3: Telegram Message Format

**New message examples**:
```
🔍 Unknown species detected at 15:42
Garden camera - Motion area: 2,847 pixels

🦔 European Hedgehog detected at 18:23  
Confidence: 87% - First sighting today!

⚠️ Motion detected at 12:15
Species identification unavailable
```

### Implementation Guidelines

**Memory Management**:
- Use `del` and `gc.collect()` after large image operations
- Monitor memory usage, skip processing if >80% RAM used
- Process images immediately, don't queue multiple captures

**Error Handling**:
- Continue operation if species ID fails
- Graceful degradation: "Motion detected - species unknown"
- Database errors should not crash main loop
- Camera errors should attempt restart

**Pi Zero Performance**:
- Longer sleep intervals during processing
- Reduce concurrent operations
- Use PIL for image resizing (lower memory than OpenCV)
- Implement processing timeouts

## File Structure Updates

```
src/
├── wildlife_detector.py      # Main orchestrator (updated from wildlife_camera.py)
├── camera_manager.py         # Enhanced for Pi Zero + YUV420 support  
├── motion_detector.py        # Minor updates (640x480, YUV420 handling)
├── species_identifier.py     # NEW: Mock implementation
├── database_manager.py       # NEW: SQLite operations
├── config.py                 # Updated Pi Zero 2 W settings
└── utils.py                  # NEW: Memory management utilities

data/
├── detections.db            # SQLite database
├── images/                  # Captured photos (timestamped)
└── logs/                    # System logs
```

## Success Criteria

After implementation:
1. ✅ Motion detection works on 640x480 YUV420 stream
2. ✅ High-res capture (1920x1080) triggered by motion  
3. ✅ Mock species identification returns "Unknown species"
4. ✅ Telegram sends text messages (not photos)
5. ✅ SQLite database logs all detections with metadata
6. ✅ System runs stable on Pi Zero 2 W for hours
7. ✅ Memory usage stays under 400MB during operation
8. ✅ Old images cleaned up automatically (max 100 stored)

## Future Integration Notes

The mock `species_identifier.py` should be designed with an interface that makes it easy to later integrate:
1. SpeciesNet cloud API wrapper
2. Google Vision API calls  
3. Local lightweight models
4. Other identification services

**Mock interface should return**:
```python
{
    'species_name': 'Unknown species',
    'confidence': 0.0,
    'api_success': False,
    'processing_time': 0.1,
    'fallback_reason': 'Mock implementation'
}
```

## Start Implementation

Please begin by updating the existing codebase according to these requirements. Start with the foundational components (config, camera manager) and build up to the full integrated system.

Focus on getting a working end-to-end system with mock species detection first, then we can integrate real species identification APIs later.