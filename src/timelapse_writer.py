"""
Lightweight timelapse / FN-audit writer (ADR-004 Phase 1).

Persists a low-rate stream of motion-resolution frames *independently of motion
triggering*, so animals that never tripped the detector (false negatives) can be
found later by a human/MegaDetector audit pass. It reuses the frame the main loop
already captures every tick — no extra camera work — and keeps disk bounded with
simple count-based rotation.
"""

import logging
import cv2

logger = logging.getLogger(__name__)


class TimelapseWriter:
    """Saves one grayscale JPEG every `timelapse_interval` seconds, capped at
    `timelapse_max_files` (oldest pruned first)."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.performance.enable_timelapse
        self.interval = config.performance.timelapse_interval
        self.max_files = config.performance.timelapse_max_files
        self.output_dir = config.storage.data_dir / "timelapse"
        self._last_capture_time = 0.0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Timelapse FN-audit enabled: every {self.interval}s → "
                f"{self.output_dir} (max {self.max_files} frames)"
            )

    def maybe_capture(self, frame, now: float) -> bool:
        """Save `frame` if the interval has elapsed. Returns True if written.

        Cheap on the common path (a single time comparison); only does disk I/O
        when an interval boundary is crossed.
        """
        if not self.enabled or frame is None:
            return False
        if now - self._last_capture_time < self.interval:
            return False

        self._last_capture_time = now
        try:
            gray = self._to_grayscale(frame)
            # Millisecond-resolution name keeps frames ordered and unique even
            # if two writes land in the same second.
            filename = self.output_dir / f"timelapse_{int(now * 1000)}.jpg"
            cv2.imwrite(str(filename), gray)
            self._prune()
            return True
        except Exception as e:
            logger.warning(f"Timelapse capture failed: {e}")
            return False

    @staticmethod
    def _to_grayscale(frame):
        """Reduce to single-channel; motion frames are already grayscale when
        color filtering is off, but handle the RGB case too."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _prune(self):
        """Delete oldest frames beyond max_files (rotation)."""
        files = sorted(self.output_dir.glob("timelapse_*.jpg"))
        excess = len(files) - self.max_files
        if excess <= 0:
            return
        for old in files[:excess]:
            try:
                old.unlink()
            except OSError as e:
                logger.warning(f"Failed to prune timelapse frame {old}: {e}")
