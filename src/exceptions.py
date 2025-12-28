"""
Consolidated exception hierarchy for the wildlife detection system.

This module provides a unified exception hierarchy that allows for:
- Consistent error handling across all components
- Hierarchical exception catching (e.g., catch all HardwareError)
- Clear categorization of error types
"""


# =============================================================================
# Base Exception
# =============================================================================

class WildlifeSystemError(Exception):
    """Base exception for all wildlife system errors."""
    pass


# =============================================================================
# Hardware Errors
# =============================================================================

class HardwareError(WildlifeSystemError):
    """Base exception for hardware-related errors."""
    pass


# Camera Errors
class CameraError(HardwareError):
    """Base exception for camera-related errors."""
    pass


class CameraInitializationError(CameraError):
    """Raised when camera initialization fails."""
    pass


class CameraOperationError(CameraError):
    """Raised when camera operations fail."""
    pass


# Database Errors
class DatabaseError(HardwareError):
    """Base exception for database errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseOperationError(DatabaseError):
    """Raised when database operations fail."""
    pass


# =============================================================================
# Processing Errors
# =============================================================================

class ProcessingError(WildlifeSystemError):
    """Base exception for processing-related errors."""
    pass


# Motion Detection Errors
class MotionDetectionError(ProcessingError):
    """Base exception for motion detection errors."""
    pass


# Species Identification Errors
class SpeciesIdentificationError(ProcessingError):
    """Base exception for species identification errors."""
    pass


class IdentificationTimeout(SpeciesIdentificationError):
    """Raised when identification times out."""
    pass


# =============================================================================
# Notification Errors
# =============================================================================

class NotificationError(WildlifeSystemError):
    """Base exception for notification-related errors."""
    pass


class TelegramError(NotificationError):
    """Base exception for Telegram-related errors."""
    pass
