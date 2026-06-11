"""
Consolidated data models for the wildlife detection system.

This module contains all dataclasses used across the system for:
- Motion detection results
- Species identification results
- Database records
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


# =============================================================================
# Detection Status taxonomy
# =============================================================================

class DetectionStatus:
    """Canonical status strings for IdentificationResult.status.

    String constants (not Enum) so they serialise trivially to/from SQLite TEXT
    and JSON without an adapter, matching the codebase's existing plain-string
    label convention (e.g. "animal", "false_positive").
    """

    #: Animal detected AND classifier confidence ≥ threshold.
    IDENTIFIED = "identified"

    #: Animal detected but classifier confidence < threshold.
    ANIMAL_UNCERTAIN = "animal_uncertain"

    #: MegaDetector found no animal (the likely-false-positive case).
    NO_ANIMAL = "no_animal"

    #: Animal-ish present but SpeciesNet returned the "no cv result" sentinel
    #: (crop unreadable / poor image quality).
    UNCLASSIFIABLE = "unclassifiable"

    #: Pipeline failure: no predictions returned, classifier exception, or a
    #: process_detection exception.  Human-readable detail in fallback_reason.
    ERROR = "error"


#: Detection statuses that are routed to human "review" (likely false
#: positives). Used by the notification layer to prepend a 🔍 REVIEW header.
#: Kept independent of the DB `gate_would_suppress` shadow column.
_REVIEW_STATUSES = frozenset({DetectionStatus.NO_ANIMAL, DetectionStatus.UNCLASSIFIABLE})


def is_review_detection(status) -> bool:
    """True if `status` is a review-class (likely false-positive) detection."""
    return status in _REVIEW_STATUSES


# =============================================================================
# Motion Detection Models
# =============================================================================

@dataclass
class MotionResult:
    """Result of motion detection analysis."""
    motion_detected: bool
    motion_area: int
    detection_confidence: float = 0.0
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    contour_count: int = 0
    # Diagnostic fields
    largest_contour_area: int = 0
    contour_areas: List[int] = field(default_factory=list)
    foreground_pixel_count: int = 0
    processing_time_ms: float = 0.0


# =============================================================================
# Species Identification Models
# =============================================================================

@dataclass
class DetectionResult:
    """Result of MegaDetector animal detection."""
    animals_detected: bool
    detection_count: int
    bounding_boxes: list  # List of dicts with bbox coords and confidence
    detections: list  # Full detection info (category, conf, bbox)
    processing_time: float


@dataclass
class IdentificationResult:
    """Result of species identification."""
    species_name: str
    confidence: float
    api_success: bool
    processing_time: float
    fallback_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Two-stage pipeline info
    detection_result: Optional[DetectionResult] = None
    animals_detected: bool = True  # For backward compatibility
    # Explicit status — replaces the overloaded "Unknown species" string.
    # Default IDENTIFIED keeps existing callers (e.g. MockSpeciesIdentifier)
    # working without modification.
    status: str = DetectionStatus.IDENTIFIED


# =============================================================================
# Database Models
# =============================================================================

@dataclass
class DetectionRecord:
    """Data class for detection records stored in the database."""
    id: Optional[int]
    timestamp: datetime
    image_path: str
    motion_area: int
    species_name: str = "Unknown species"
    confidence_score: float = 0.0
    processing_time: float = 0.0
    api_success: bool = False
