"""Simplexity logger.

This module provides the main logger instance for the simplexity package.
It configures Python's warnings system to be captured by the logging system
and creates a logger instance named "simplexity" for use throughout the package.
"""

import logging

# Configure Python's warnings system to be captured by the logging system.
# This ensures that warnings issued by the warnings module are redirected to
# the logging system, allowing them to be handled consistently with other
# log messages. This is a module-level side effect that occurs on import.
logging.captureWarnings(True)

# Main logger instance for the simplexity package.
# This logger is used throughout the codebase for info, debug, warning, and error messages.
# It can be imported and used directly: `from simplexity.logger import SIMPLEXITY_LOGGER`
SIMPLEXITY_LOGGER: logging.Logger = logging.getLogger("simplexity")
