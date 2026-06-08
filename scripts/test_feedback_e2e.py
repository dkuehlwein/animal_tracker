#!/usr/bin/env python3
"""
End-to-end manual test for the Phase-1 human-feedback loop (ADR-004).

Mirrors the real send path (`wildlife_system.send_notification`, single-image
branch): logs a detection row, then sends the photo to Telegram with the
inline feedback keyboard whose callback_data embeds that detection_id.

Pair this with the sidecar to close the loop:

    # terminal 1 — listen for taps
    uv run python src/telegram_feedback.py

    # terminal 2 — send a test notification
    uv run python scripts/test_feedback_e2e.py [path/to/image.jpg]

Then tap a button in Telegram and re-run with --check (or query the DB) to see
the recorded row.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("src")

from config import Config
from database_manager import DatabaseManager
from feedback_protocol import build_feedback_keyboard
from notification_service import NotificationService


def _latest_image(config: Config) -> Path:
    images = sorted(
        config.storage.image_dir.glob(f"{config.storage.image_prefix}*.jpg"),
        key=lambda p: p.stat().st_mtime,
    )
    if not images:
        raise SystemExit(f"No images found in {config.storage.image_dir}")
    return images[-1]


async def send(config: Config, database: DatabaseManager, image_path: Path) -> int:
    detection_id = database.log_detection(
        image_path=str(image_path),
        motion_area=1234,
        species_name="E2E feedback test",
        confidence_score=0.99,
        api_success=True,
        animals_detected=True,
    )
    print(f"Logged detection id={detection_id} for {image_path.name}")

    service = NotificationService(config)
    keyboard = build_feedback_keyboard(detection_id)
    caption = (
        f"🧪 E2E feedback test — detection #{detection_id}\n"
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "Tap a button below; it should appear in detection_feedback."
    )
    ok = await service.send_photo_with_caption(image_path, caption, reply_markup=keyboard)
    print(f"send_photo_with_caption -> {ok}")
    return detection_id


def check(database: DatabaseManager, detection_id: int | None) -> None:
    import sqlite3

    conn = sqlite3.connect(database.db_path)
    if detection_id is None:
        rows = conn.execute(
            "SELECT id, detection_id, label, source, created_at "
            "FROM detection_feedback ORDER BY id DESC LIMIT 10"
        ).fetchall()
    else:
        rows = database.get_feedback(detection_id)
    if not rows:
        print("No feedback rows yet.")
    else:
        print("Feedback rows:")
        for r in rows:
            print(" ", r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Image to send (default: latest capture)")
    parser.add_argument("--check", action="store_true", help="Only show recorded feedback rows")
    parser.add_argument("--detection-id", type=int, help="Filter --check to one detection")
    args = parser.parse_args()

    config = Config()
    database = DatabaseManager(config)

    if args.check:
        check(database, args.detection_id)
        return

    image_path = Path(args.image) if args.image else _latest_image(config)
    detection_id = asyncio.run(send(config, database, image_path))
    print(
        f"\nSent. Now tap a button in Telegram, then run:\n"
        f"  uv run python scripts/test_feedback_e2e.py --check --detection-id {detection_id}"
    )


if __name__ == "__main__":
    main()
