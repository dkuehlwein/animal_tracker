# Wildlife Camera System

A Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, identifies species using AI, and sends notifications to a Telegram channel.

## Features

- **Motion Detection**: OpenCV-based motion detection with central region weighting
- **AI Species Identification**: Google SpeciesNet for wildlife identification (2000+ species)
- **Geographic Filtering**: Species predictions filtered for Bonn, Germany
- **Automatic Photo Capture**: High-resolution images when motion is detected
- **Real-time Notifications**: Telegram notifications with species information
- **Database Logging**: SQLite database tracking all detections
- **Configurable System**: Environment-based configuration for all parameters

## Requirements

- **Hardware**:
  - Raspberry Pi 5 with 8GB RAM
  - Raspberry Pi Camera Module
  - Adequate storage for images and models (~2GB recommended)

- **Software**:
  - Python 3.7+
  - Raspberry Pi OS (64-bit recommended)
  - Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd animal_tracker
```

2. Install system dependencies (if needed):
```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev libopenjp2-7
```

3. Install Python packages:
```bash
pip install -r requirements.txt
```

4. Download SpeciesNet models (first time only, ~500MB-1GB):
```bash
python -c "from speciesnet.ensemble import SpeciesNetEnsemble; SpeciesNetEnsemble(country='DEU')"
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Telegram bot token and chat ID
```

### Getting Your Telegram Group Chat ID

To send notifications to a Telegram group:

1. **Create a bot** via [@BotFather](https://t.me/BotFather):
   - Send `/newbot` and follow instructions
   - Save the bot token

2. **Disable Privacy Mode**:
   - Send `/setprivacy` to @BotFather
   - Select your bot
   - Choose "Disable"

3. **Get your group chat ID**:
   - Add [@JSONDumpBot](https://t.me/JSONDumpBot) to your group
   - Forward any message from your group to @JSONDumpBot
   - It will show the chat ID (e.g., `-1001234567890`)
   - Remove @JSONDumpBot from the group

4. **Add your bot to the group**:
   - Add your bot to the group
   - Optionally make it an administrator for reliable message delivery

5. **Update your `.env` file**:
   ```bash
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   TELEGRAM_CHAT_ID=-1001234567890
   ```

## Usage

### Camera Setup and Focus Adjustment

Before running the system, you should adjust the camera focus and positioning:

1. Run the camera preview server:
```bash
python3 scripts/camera_preview.py
```

2. Open a web browser and navigate to `http://<raspberry-pi-ip>:8000`

3. Adjust the camera lens focus ring while watching the live preview:
   - For distant objects (wildlife at a distance), rotate towards infinity focus
   - Fine-tune the focus for your specific target area (e.g., bird feeder, squirrel house)
   - The preview shows full 1920x1080 resolution for precise focus checking

4. Press Ctrl+C in the terminal to stop the preview server

### Running the Detection System

**Manual Mode:**

Run the wildlife detection system directly:
```bash
python src/wildlife_system.py
```

**Auto-start on Boot (Recommended):**

Set up the system as a systemd service to automatically start on boot and restart on crashes:

1. Install the service:
```bash
sudo cp wildlife-camera.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wildlife-camera.service
sudo systemctl start wildlife-camera.service
```

2. Manage the service:
```bash
# Check service status
sudo systemctl status wildlife-camera.service

# View logs
sudo systemctl status wildlife-camera.service  # Recent logs
sudo journalctl -u wildlife-camera.service -n 50 -f  # Follow logs

# Stop/start/restart
sudo systemctl stop wildlife-camera.service
sudo systemctl start wildlife-camera.service
sudo systemctl restart wildlife-camera.service

# Disable auto-start
sudo systemctl disable wildlife-camera.service
```

The system will automatically:
- Monitor for motion using low-resolution frames (640x480)
- Capture high-resolution photos when motion is detected (1920x1080)
- Identify species using SpeciesNet AI
- Send Telegram notifications with species information
- Store photos in the `data/images` directory
- Log all detections to SQLite database
- Restart automatically on crashes (when running as service)

## Configuration

The system is configured via environment variables in `.env` file:

**Required:**
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Optional (with defaults):**
```bash
# Motion Detection
MOTION_THRESHOLD=2000
MOTION_CONSECUTIVE_REQUIRED=2
MOTION_FRAME_INTERVAL=0.2

# Species Identification
SPECIES_COUNTRY_CODE=DEU
SPECIES_REGION=NW
SPECIES_MIN_DETECTION_CONF=0.6
SPECIES_MIN_CLASS_CONF=0.5
SPECIES_UNKNOWN_THRESHOLD=0.5

# Performance
PERFORMANCE_COOLDOWN=30.0
PERFORMANCE_MAX_IMAGES=100

# Camera
CAMERA_MAIN_RESOLUTION=1920x1080
CAMERA_MOTION_RESOLUTION=640x480
```

See `src/config.py` for all available configuration options.

## Directory Structure

```
animal_tracker/
├── src/
│   ├── wildlife_system.py       # Main system orchestrator
│   ├── config.py                # Configuration management
│   ├── camera_manager.py        # Camera operations
│   ├── motion_detector.py       # Motion detection
│   ├── species_identifier.py    # SpeciesNet integration
│   ├── database_manager.py      # SQLite database
│   ├── notification_service.py  # Telegram notifications
│   ├── resource_manager.py      # Memory, storage, system monitoring
│   ├── models.py                # Data models (MotionResult, DetectionRecord, etc.)
│   ├── exceptions.py            # Exception hierarchy
│   └── utils.py                 # Utilities (PerformanceTimer, SharpnessAnalyzer, etc.)
├── scripts/
│   └── camera_preview.py        # Live camera preview for focus adjustment
├── data/
│   ├── images/                  # Captured photos
│   ├── logs/                    # System logs
│   └── detections.db            # Detection database
├── tests/                       # Unit tests
├── docs/                        # Documentation
└── requirements.txt             # Python dependencies
```

## Species Identification

The system uses **Google SpeciesNet**, a state-of-the-art AI model specifically designed for camera trap wildlife identification:

- **Two-Stage Pipeline**: 
  - Stage 1: MegaDetector finds animals in images (categories: animal, person, vehicle)
  - Stage 2: SpeciesNet classifies detected animals to species level
- **Accuracy**: 94.5% species-level identification
- **Coverage**: 2000+ species labels
- **Geographic Filtering**: Automatically filters to species found in your region
- **Local Processing**: Runs entirely on Raspberry Pi 5 (no cloud required)
- **Performance**: ~15-20 seconds per image on Pi 5 (detection + classification)

**Configured for Bonn, Germany** (Country: DEU, Region: NW)

**Species Name Format**: SpeciesNet returns taxonomic paths in the format:
```
UUID;class;order;family;genus;species;common_name
```
Example: `e4d1e892-0e4b-475a-a8ac-b5c3502e0d55;mammalia;rodentia;sciuridae;;;sciuridae family`

The system extracts the common name (last field) for display in notifications.

Common expected species:
- European Hedgehog (*Erinaceus europaeus*) - Eurasian hedgehog
- Red Fox (*Vulpes vulpes*) - Red fox  
- European Robin (*Erithacus rubecula*) - European robin
- Domestic Cat (*Felis catus*) - Domestic cat
- Red Squirrel (*Sciurus vulgaris*) - Eurasian red squirrel
- Various garden birds

## Troubleshooting

**Camera Issues:**
1. Ensure the camera module is properly connected
2. Check camera permissions: `sudo usermod -a -G video $USER`
3. Test camera: `libcamera-hello`
4. Adjust focus using the preview script: `python3 scripts/camera_preview.py`
5. For blurry images at distance, rotate focus ring to infinity (opposite end from minimum focus)

**SpeciesNet Issues:**
1. Ensure models are downloaded (first run takes time)
2. Check available memory: `free -h` (need ~3GB free)
3. Check model cache: `~/.cache/speciesnet/`

**Configuration Issues:**
1. Verify Telegram bot token and chat ID are correct
2. Check `.env` file exists and is properly formatted
3. Review logs in `data/logs/`

**Performance Issues:**
1. Reduce `SPECIES_PROCESSING_TIMEOUT` if inference is slow
2. Increase cooldown period to reduce processing load
3. Monitor system resources with `htop`

## Network Watchdog

The system includes a network watchdog service that monitors WiFi connectivity and takes corrective action on failures. This is especially useful for remote/outdoor deployments where the Pi may lose WiFi connection.

### Features

- Monitors connectivity every 30 seconds (pings gateway, Google DNS, Cloudflare DNS)
- Automatic recovery actions on failure:
  - After 3 failures: Restarts WiFi interface
  - After 5 failures: Restarts NetworkManager
  - After 8 failures: Reboots system (last resort)
- Telegram notifications for:
  - System startup (includes IP address and signal strength)
  - Connectivity loss/restoration
  - Recovery actions taken
- Logs to systemd journal and `/var/log/network-watchdog.log`

### Installation

```bash
# Install the service
sudo cp network-watchdog.service /etc/systemd/system/
sudo cp configs/99-wifi-stability.conf /etc/NetworkManager/conf.d/
sudo systemctl daemon-reload
sudo systemctl enable network-watchdog.service
sudo systemctl start network-watchdog.service
```

### Management

```bash
# Check status
sudo systemctl status network-watchdog

# View logs
journalctl -u network-watchdog -f

# Restart after config changes
sudo systemctl restart network-watchdog
```

### Configuration

Edit `scripts/network_watchdog.sh` to adjust:
- `CHECK_INTERVAL`: Seconds between connectivity checks (default: 30)
- `MAX_FAILURES_BEFORE_RESTART`: Failures before WiFi restart (default: 3)
- `MAX_FAILURES_BEFORE_REBOOT`: Failures before system reboot (default: 8)

Telegram notifications are automatically enabled if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in your `.env` file.

## Power Consumption

Running 24/7 on a Raspberry Pi 5 (8GB):

| State | Power Draw |
|-------|-----------|
| Idle (WiFi on) | ~3-4W |
| Motion detection | ~5-6W |
| SpeciesNet inference | ~8-10W |
| **Average** | **~5W** |

### Annual Cost Estimate

| Period | Consumption | Cost (€0.30-0.40/kWh) |
|--------|-------------|----------------------|
| Daily | 0.12 kWh | €0.04-0.05 |
| Monthly | 3.6 kWh | €1.08-1.44 |
| **Yearly** | **43.8 kWh** | **€13-18** |
