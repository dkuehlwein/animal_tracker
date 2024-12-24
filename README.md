# Wildlife Camera System

A Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, and sends them to a Telegram channel.

## Features

- Motion detection using OpenCV
- Automatic photo capture when motion is detected
- Real-time notifications via Telegram
- Configurable motion detection sensitivity
- Automatic image storage with timestamps

## Requirements

- Raspberry Pi 5
- Raspberry Pi Camera Module
- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd animal_tracker
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your camera module is properly connected to the Raspberry Pi
2. Run the wildlife camera system:
```bash
python src/wildlife_camera.py
```

The system will automatically:
- Monitor for motion
- Capture photos when motion is detected
- Send captured photos to the configured Telegram channel
- Store photos in the `data` directory

## Configuration

The system uses the following default parameters:
- Motion threshold: 25
- Minimum contour area: 500
- Image resolution: 640x480

These parameters can be adjusted in the `wildlife_camera.py` file.

## Directory Structure

- `/src` - Source code
- `/data` - Captured images
- `/docs` - Documentation
- `/test` - Test code

## Troubleshooting

If you encounter issues:
1. Ensure the camera module is properly connected
2. Verify Telegram bot token and chat ID are correct
3. Check camera permissions
4. Ensure all dependencies are properly installed
