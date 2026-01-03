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
