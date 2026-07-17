"""Scene-unchanged gate: frame comparator + rolling empty-scene reference set.

Compares a candidate frame against recent known-empty reference frames so
review-class detections whose scene is near-identical to a recently-seen
empty background can be muted upstream (wiring happens in later tasks — this
module is a standalone comparator + reference set only).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Frames are downsized to this common size before comparison — small enough
# to be cheap, large enough to preserve coarse scene structure.
_COMPARE_SIZE = (64, 64)

# Mean-abs-diff of two zero-mean/unit-std normalized images is clamped to this
# value before being mapped to a [0, 1] similarity score.
_MAX_CLAMPED_DIFF = 2.0


def _normalize(image: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-ish-contrast normalize a grayscale image (float64)."""
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + 1e-6)


def compute_scene_similarity(
    frame_path: Union[str, Path], reference_path: Union[str, Path]
) -> Optional[float]:
    """Compare two frames for scene similarity, robust to exposure/AE shifts.

    Loads both images as grayscale, resizes to a small common size, and
    normalizes each to zero-mean/unit-ish contrast (subtract mean, divide by
    std with epsilon) so global exposure/auto-exposure shifts don't dominate
    the comparison. Similarity is `1.0 - clamped mean-abs-diff` mapped to
    [0, 1], where 1.0 means identical scenes.

    Returns None if either image can't be read or any error occurs during
    comparison (logged at WARNING). Never raises.
    """
    try:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            logger.warning(f"compute_scene_similarity: could not read frame image: {frame_path}")
            return None

        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            logger.warning(
                f"compute_scene_similarity: could not read reference image: {reference_path}"
            )
            return None

        frame_small = cv2.resize(frame, _COMPARE_SIZE).astype(np.float64)
        reference_small = cv2.resize(reference, _COMPARE_SIZE).astype(np.float64)

        frame_norm = _normalize(frame_small)
        reference_norm = _normalize(reference_small)

        mean_abs_diff = float(np.mean(np.abs(frame_norm - reference_norm)))
        clamped_diff = min(mean_abs_diff, _MAX_CLAMPED_DIFF)
        similarity = 1.0 - (clamped_diff / _MAX_CLAMPED_DIFF)

        return max(0.0, min(1.0, similarity))

    except Exception as e:
        logger.warning(
            f"compute_scene_similarity failed for frame={frame_path} "
            f"reference={reference_path}: {e}"
        )
        return None


class SceneReferenceSet:
    """Rolling set of recent known-empty reference frames for scene comparison.

    Bounded by both a maximum count (`max_refs`) and a maximum age
    (`max_age_hours`), evicted lazily on `.add()` and `.best_similarity()`.
    """

    def __init__(self, max_refs: int, max_age_hours: float):
        self.max_refs = max_refs
        self.max_age_hours = max_age_hours
        self._refs: List[Tuple[str, datetime]] = []

    def add(self, image_path: Union[str, Path], timestamp: datetime) -> None:
        """Append a reference frame, then evict oldest-beyond-max_refs and
        anything older than max_age_hours relative to this add's timestamp."""
        self._refs.append((str(image_path), timestamp))
        self._refs.sort(key=lambda entry: entry[1])
        self._prune_by_age(timestamp)
        if len(self._refs) > self.max_refs:
            self._refs = self._refs[-self.max_refs:]

    def best_similarity(self, frame_path: Union[str, Path], now: datetime) -> Optional[float]:
        """Prune stale refs against `now`, then return the max similarity of
        `frame_path` against remaining refs (skipping failed comparisons).

        Returns None if there are no refs (before or after pruning) or every
        comparison failed.
        """
        self._prune_by_age(now)
        if not self._refs:
            return None

        best: Optional[float] = None
        for ref_path, _ in self._refs:
            similarity = compute_scene_similarity(frame_path, ref_path)
            if similarity is None:
                continue
            if best is None or similarity > best:
                best = similarity
        return best

    def seed(self, rows: List[Tuple[Union[str, Path], datetime]]) -> None:
        """Bulk-load reference rows (oldest -> newest), same eviction rules
        as `.add()`. Rows whose path doesn't exist on disk are skipped."""
        for image_path, timestamp in rows:
            if not Path(image_path).exists():
                continue
            self.add(image_path, timestamp)

    def _prune_by_age(self, reference_time: datetime) -> None:
        max_age = timedelta(hours=self.max_age_hours)
        self._refs = [
            (path, ts) for path, ts in self._refs if reference_time - ts <= max_age
        ]
