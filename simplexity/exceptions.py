"""Custom exception hierarchy for the simplexity package."""


class SimplexityException(Exception):
    """Base exception for Simplexity."""


class ConfigValidationError(SimplexityException):
    """Exception raised when a config is invalid."""


class DeviceResolutionError(SimplexityException):
    """Exception raised when a device resolution fails."""
