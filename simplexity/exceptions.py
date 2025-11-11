class SimplexityException(Exception):
    """Base exception for Simplexity."""

    pass


class ConfigValidationError(SimplexityException):
    """Exception raised when a config is invalid."""

    pass


class DeviceResolutionError(SimplexityException):
    """Exception raised when a device resolution fails."""

    pass
