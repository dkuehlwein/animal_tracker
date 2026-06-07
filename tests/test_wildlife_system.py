"""
Integration test for wildlife_system main-loop behavior.

Verifies that the post-detection cooldown does NOT starve MOG2 — i.e.
motion_detector.detect() must keep being called every tick during the
cooldown window so the background model can track scene drift.
"""

import asyncio
import sqlite3
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

sys.path.append('src')


@pytest.fixture
def system(monkeypatch, tmp_path):
    """A WildlifeSystem with an isolated temp DB and no real telegram/camera."""
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('PERFORMANCE_ENABLE_TIMELAPSE', 'false')
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from database_manager import DatabaseManager

    sys_obj = WildlifeSystem()
    # Isolate the database to a temp file.
    cfg = SimpleNamespace(storage=SimpleNamespace(database_path=str(tmp_path / "d.db")))
    sys_obj.database = DatabaseManager(cfg)
    sys_obj.reference_frame = None  # skip frame-stability imread
    return sys_obj


def _identification(animals_detected, boxes=None):
    from data_models import IdentificationResult, DetectionResult
    boxes = boxes or []
    det = DetectionResult(
        animals_detected=animals_detected,
        detection_count=len(boxes),
        bounding_boxes=boxes,
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="Fox" if animals_detected else "Unknown species",
        confidence=0.9 if animals_detected else 0.0,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=animals_detected,
    )


def test_process_detection_persists_richer_fields_and_id(system):
    from data_models import MotionResult
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}, {'confidence': 0.85}])
    )
    motion = MotionResult(
        motion_detected=True, motion_area=5000, contour_count=3,
        largest_contour_area=2200, foreground_pixel_count=3300,
    )

    result, ts = system.process_detection("capture.jpg", 5000, motion)

    assert result['detection_id'] is not None
    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['animals_detected'] == 1
    assert row['detection_count'] == 2
    assert row['max_detection_confidence'] == pytest.approx(0.85)
    assert row['contour_count'] == 3
    assert row['largest_contour_area'] == 2200
    assert row['foreground_pixel_count'] == 3300
    assert row['gate_would_suppress'] == 0  # animal present → would NOT suppress


def test_process_detection_shadow_gate_records_suppression(system):
    """No animal → gate would suppress, but the row is still written (shadow mode)."""
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(False)
    )
    result, _ = system.process_detection("capture.jpg", 800, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['gate_would_suppress'] == 1
    assert row['animals_detected'] == 0


@pytest.mark.asyncio
async def test_send_notification_attaches_feedback_keyboard(system, tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.telegram_service = MagicMock()
    system.telegram_service.send_photo_with_caption = AsyncMock()
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0

    species_result = {'species_name': 'Fox', 'confidence': 0.9, 'animals_detected': True,
                      'detection_id': 123, 'detection_result': None}
    await system.send_notification(species_result, 5000, datetime.now(), image_path=img)

    _, kwargs = system.telegram_service.send_photo_with_caption.call_args
    keyboard = kwargs['reply_markup']
    data = [b.callback_data for b in keyboard.inline_keyboard[0]]
    assert data == ["fb:123:a", "fb:123:fp", "fb:123:ws"]


@pytest.mark.asyncio
async def test_cooldown_keeps_feeding_motion_detector(monkeypatch):
    """During post-detection cooldown the loop must keep calling
    motion_detector.detect() so MOG2 stays calibrated to the live scene.

    With the original (broken) cooldown gate, detect() is called exactly
    once: the first tick triggers a detection, sets last_detection_time,
    and every subsequent tick hits the early-`continue` and skips
    detect() entirely. After the fix detect() is called every tick,
    even while cooldown suppresses the heavy capture+ID+notify path.
    """
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('MOTION_FRAME_INTERVAL', '0.001')
    monkeypatch.setenv('PERFORMANCE_IDLE_SLEEP', '0')
    monkeypatch.setenv('PERFORMANCE_COOLDOWN_SLEEP', '0')
    monkeypatch.setenv('PERFORMANCE_DAYLIGHT_ONLY', 'false')
    monkeypatch.setenv('PERFORMANCE_COOLDOWN_PERIOD', '30')

    # Force config + wildlife_system reload so env vars are picked up fresh
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from data_models import MotionResult

    system = WildlifeSystem()

    detect_calls = []

    def detect_side_effect(frame):
        detect_calls.append(frame)
        # Tick 1 returns motion (arms cooldown). Subsequent ticks: no motion.
        first = len(detect_calls) == 1
        return MotionResult(
            motion_detected=first,
            motion_area=5000 if first else 0,
        )

    # Replace motion_detector with a mock we can observe
    system.motion_detector = MagicMock()
    system.motion_detector.detect.side_effect = detect_side_effect
    system.motion_detector.is_warming_up = False
    system._was_warming_up = False

    # Replace camera with a mock that yields a fake frame and a no-op session
    fake_frame = np.zeros((480, 640), dtype=np.uint8)
    system.camera = MagicMock()
    system.camera.capture_motion_frame.return_value = fake_frame
    system.camera.consume_restart_flag.return_value = False
    system.camera.capture_high_res_frame.return_value = None
    system.camera.capture_burst_frames.return_value = []  # forces image_path=None
    # MagicMock's default __enter__/__exit__ make camera_session() usable as a CM

    # Replace remaining components so the loop has no real side effects
    system.telegram_service = MagicMock()
    system.telegram_service.send_text_message = AsyncMock()
    system.telegram_service.send_photo_with_caption = AsyncMock()
    system.telegram_service.send_media_group = AsyncMock()
    system.telegram_service.send_detection_notification = AsyncMock()

    system.system_monitor = MagicMock()
    system.system_monitor.should_skip_processing.return_value = False
    system.system_monitor.memory_manager = MagicMock()

    system.sun_checker = MagicMock()
    system.sun_checker.is_daytime.return_value = True

    # Run the loop briefly, then cancel via timeout
    try:
        await asyncio.wait_for(system.run(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    # Cooldown is 30s, the loop ran for 0.3s — every tick after the first
    # is inside cooldown. detect() must still be called on every tick.
    assert len(detect_calls) >= 5, (
        f"Cooldown is starving MOG2: detect() called only {len(detect_calls)} "
        f"times in 0.3s. Expected >=5."
    )
