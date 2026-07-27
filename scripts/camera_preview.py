#!/usr/bin/env python3
"""
Camera preview MJPEG server for focus and positioning adjustment.
Based on official Picamera2 example.
Access at http://<raspberry-pi-ip>:8000
"""

import io
import logging
import os
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Stream tuning (env-overridable). Defaults are sized to fit a weak/slow Wi-Fi
# uplink: a full-HD, uncapped MJPEG stream is tens of Mbit/s and saturates a
# marginal link, causing severe lag/bufferbloat. 720p @ 10fps @ q70 is ~3-5
# Mbit/s and stays sharp enough for focus/positioning. Bump these if the link
# is good (e.g. on Ethernet): PREVIEW_WIDTH/HEIGHT, PREVIEW_FPS, PREVIEW_QUALITY.
PREVIEW_WIDTH = int(os.environ.get("PREVIEW_WIDTH", "1280"))
PREVIEW_HEIGHT = int(os.environ.get("PREVIEW_HEIGHT", "720"))
PREVIEW_FPS = int(os.environ.get("PREVIEW_FPS", "10"))
PREVIEW_QUALITY = int(os.environ.get("PREVIEW_QUALITY", "70"))

PAGE = """\
<html>
<head>
<title>Wildlife Camera Preview</title>
<style>
    body {
        margin: 0;
        padding: 20px;
        background-color: #1a1a1a;
        color: #fff;
        font-family: Arial, sans-serif;
    }
    h1 {
        text-align: center;
    }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    img {
        max-width: 100%;
        height: auto;
        border: 2px solid #4a4a4a;
        border-radius: 5px;
    }
</style>
</head>
<body>
<div class="container">
<h1>Wildlife Camera Preview - Adjust Focus & Position</h1>
<img src="stream.mjpg" />
<p>Press Ctrl+C in terminal to stop</p>
</div>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Initialize camera. Resolution/framerate are capped to fit the Wi-Fi uplink
# (see PREVIEW_* above) so the MJPEG stream stays smooth on a marginal link.
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)},
    controls={"FrameRate": PREVIEW_FPS},
    buffer_count=2,  # smaller buffer = lower capture-to-display latency
))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(q=PREVIEW_QUALITY), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("=" * 60)
    print("Wildlife Camera Preview Server Starting")
    print("=" * 60)
    print("Access the preview at: http://192.168.2.141:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    server.serve_forever()
finally:
    picam2.stop_recording()
