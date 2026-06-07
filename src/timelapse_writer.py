"""
Lightweight timelapse / FN-audit writer (ADR-004 Phase 1).

Persists a low-rate stream of motion-resolution frames *independently of motion
triggering*, so animals that never tripped the detector (false negatives) can be
found later by a human/MegaDetector audit pass. It reuses the frame the main loop
already captures every tick — no extra camera work — and keeps disk bounded with
simple count-based rotation.

Rotation is tracked in an in-memory deque so the hot path never globs/sorts the
(potentially ~10k-file) directory; the only per-write cost is a single grayscale
JPEG encode.
"""

import logging
from collections import deque

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
        # Oldest-first record of on-disk frames; seeded from any existing files
        # so rotation survives a restart without re-globbing on every write.
        self._files = deque()

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._files.extend(sorted(self.output_dir.glob("timelapse_*.jpg")))
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
            path = self.output_dir / f"timelapse_{int(now * 1000)}.jpg"
            cv2.imwrite(str(path), gray)
            self._files.append(path)
            self._prune()
            return True
        except Exception as e:
            logger.warning(f"Timelapse capture failed: {e}")
            return False

    @staticmethod
    def _to_grayscale(frame):
        """Reduce to single-channel; motion frames are already grayscale when
        color filtering is off, but handle the 3-channel case too."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _prune(self):
        """Delete oldest frames beyond max_files (O(1) amortized via the deque)."""
        while len(self._files) > self.max_files:
            old = self._files.popleft()
            try:
                old.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Failed to prune timelapse frame {old}: {e}")
