class SimplexityException(Exception):
    """Base exception for Simplexity."""

    pass


class ConfigValidationError(SimplexityException):
    """Exception raised when a config is invalid."""

    pass
