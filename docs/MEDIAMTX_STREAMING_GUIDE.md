# MediaMTX Streaming Guide for Wildlife Camera

MediaMTX is a zero-dependency media server perfect for adding live streaming capabilities to your wildlife camera system. This guide covers installation and setup for Raspberry Pi Zero 2 W compatibility.

## Pi Zero 2 W Compatibility

**✅ MediaMTX works on Pi Zero 2 W**

- **Architecture**: Quad-core 64-bit ARM Cortex-A53 @ 1GHz
- **Memory**: 512MB LPDDR2 RAM
- **OS Support**: Both 32-bit and 64-bit Raspberry Pi OS
- **Confirmed Working**: Users successfully running MediaMTX v1.8.5+ on Pi Zero 2 W with Camera Module 3

### Architecture Selection

- **32-bit OS**: Use `armv7` binary
- **64-bit OS**: Use `arm64v8` binary
- **Check your architecture**: `uname -m`

## Installation Instructions

### Step 1: Determine Your Architecture
```bash
uname -m
```
- If output is `armv7l`: Use armv7 binary
- If output is `aarch64`: Use arm64v8 binary

### Step 2: Download MediaMTX
```bash
# For 32-bit (armv7)
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.8.5_linux_armv7.tar.gz

# For 64-bit (arm64v8)
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.8.5_linux_arm64v8.tar.gz
```

### Step 3: Extract and Setup
```bash
# Extract the archive
tar -xzf mediamtx_*.tar.gz --one-top-level=mediamtx

# Move to installation directory
cd mediamtx

# Make binary executable
chmod +x mediamtx
```

### Step 4: Configure for Pi Camera
Edit `mediamtx.yml` to add Pi camera support:

```yaml
paths:
  cam:
    runOnInit: rpicam-vid -t 0 --width 1920 --height 1080 --inline -o - | ffmpeg -f h264 -i - -c copy -f rtsp rtsp://localhost:8554/cam
    runOnInitRestart: yes
```

### Step 5: Run MediaMTX
```bash
# Start MediaMTX
./mediamtx
```

## Integration with Wildlife Camera System

### Option 1: Separate Stream
Run MediaMTX alongside your wildlife camera system:

```bash
# In one terminal - your wildlife camera
python src/wildlife_camera.py

# In another terminal - MediaMTX streaming
cd mediamtx && ./mediamtx
```

### Option 2: Shared Camera Access
Modify your camera manager to provide a streaming endpoint:

```python
# Add to camera_manager.py
def start_preview_stream(self):
    """Start a preview stream for setup/monitoring"""
    if not self.is_camera_active:
        self.picam2.start_recording(Encoder(), FileOutput("/tmp/stream.h264"))
```

## Viewing the Stream

### RTSP (Traditional Players)
```
rtsp://your-pi-ip:8554/cam
```
- Use VLC Media Player: Media → Open Network Stream
- Use mpv: `mpv rtsp://192.168.1.100:8554/cam`

### WebRTC (Browsers)
```
http://your-pi-ip:8889/cam
```
- Low latency (~300ms)
- Works in modern browsers
- No additional software needed

### HLS (HTTP Live Streaming)
```
http://your-pi-ip:8888/cam
```
- Good for web integration
- Higher latency but better compatibility

## Performance Considerations for Pi Zero 2 W

### Memory Optimization
- **32-bit OS recommended**: Provides ~240MB available RAM vs ~99MB on 64-bit
- Monitor memory usage: `free -h`
- Consider reducing stream resolution for better performance

### Stream Settings for Pi Zero 2 W
```yaml
# Optimized configuration for Pi Zero 2 W
paths:
  cam:
    runOnInit: rpicam-vid -t 0 --width 1280 --height 720 --framerate 15 --inline -o - | ffmpeg -f h264 -i - -c copy -f rtsp rtsp://localhost:8554/cam
    runOnInitRestart: yes
```

### Power Requirements
- Use quality 2.5A power supply
- Consider adding heatsink for sustained operation
- Monitor temperature: `vcgencmd measure_temp`

## Systemd Service (Optional)

Create a service to auto-start MediaMTX:

```bash
# Create service file
sudo nano /etc/systemd/system/mediamtx.service
```

```ini
[Unit]
Description=MediaMTX streaming server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/mediamtx
ExecStart=/home/pi/mediamtx/mediamtx
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable mediamtx
sudo systemctl start mediamtx
```

## Troubleshooting

### Common Issues

**"Illegal instruction" error**:
- Wrong architecture binary downloaded
- Verify with `uname -m` and download correct version

**Camera busy error**:
- Another process is using the camera
- Stop wildlife camera system: `pkill -f wildlife_camera.py`
- Or configure shared access

**High CPU usage**:
- Reduce resolution and framerate
- Use hardware encoding when available
- Monitor with `htop`

**Network connectivity issues**:
- Check firewall: `sudo ufw status`
- Verify ports 8554 (RTSP), 8888 (HLS), 8889 (WebRTC) are open
- Use `netstat -tlnp` to verify MediaMTX is listening

### Latency Optimization

For minimal latency setup:
1. Use WebRTC viewer (http://pi-ip:8889/cam)
2. Reduce resolution to 1280x720 or lower
3. Increase framerate to 30fps if Pi can handle it
4. Use wired ethernet instead of Wi-Fi when possible

## Security Considerations

### Access Control
MediaMTX supports authentication:

```yaml
# Add to mediamtx.yml
authMethods: [internal]
paths:
  cam:
    publishUser: admin
    publishPass: your-secure-password
    readUser: viewer
    readPass: viewer-password
```

### Firewall Rules
```bash
# Allow specific IPs only
sudo ufw allow from 192.168.1.0/24 to any port 8554
sudo ufw allow from 192.168.1.0/24 to any port 8889
```

## Advanced Configuration

### Multiple Camera Streams
```yaml
paths:
  main_cam:
    runOnInit: rpicam-vid -t 0 --width 1920 --height 1080 -o - | ffmpeg -f h264 -i - -c copy -f rtsp rtsp://localhost:8554/main_cam
  preview_cam:
    runOnInit: rpicam-vid -t 0 --width 640 --height 480 -o - | ffmpeg -f h264 -i - -c copy -f rtsp rtsp://localhost:8554/preview_cam
```

### Recording Streams
```yaml
paths:
  cam:
    record: yes
    recordPath: /home/pi/recordings/%Y-%m-%d_%H-%M-%S.mp4
    recordDeleteAfter: 24h
```

This setup provides a robust streaming solution for your wildlife camera, allowing real-time monitoring during setup and ongoing surveillance capabilities.