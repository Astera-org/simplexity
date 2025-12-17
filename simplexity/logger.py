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


def add_handlers_to_existing_loggers() -> None:
    """Add root logger's handlers to existing loggers that don't propagate.

    This is useful for loggers created before fileConfig() that have propagate=0
    or otherwise don't inherit handlers from root. Most loggers propagate to root
    by default, so they'll use root's handlers automatically.

    This function adds ALL handlers from root (not just file handlers) to ensure
    consistency for loggers that need explicit handlers.

    **When this is useful:**
    - Loggers with propagate=0 created before fileConfig() runs (they won't inherit
      root's handlers automatically)
    - Third-party loggers that disable propagation and were created during early imports
      (e.g., jax._src.xla_bridge if it has propagate=0)

    **When it's NOT needed:**
    - Most loggers propagate to root by default, so they automatically use root's handlers
    - fileConfig() with disable_existing_loggers=False should update existing loggers
      that are specified in the INI config

    **Recommendation:**
    Test without calling this function first. If you find loggers that should be
    logging to the file but aren't (especially those with propagate=0), then call
    this function after configure_logging_from_file(). Otherwise, it may be unnecessary.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        return

    # Add all root handlers to loggers that don't propagate and don't already have them
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        # Skip root logger itself
        if logger is root_logger:
            continue

        # Only add handlers to loggers that don't propagate (they need their own handlers)
        # or loggers that were created before fileConfig and might not have handlers
        if not logger.propagate or not logger.handlers:
            for handler in root_logger.handlers:
                # Check if logger already has this exact handler object (by identity, not similarity)
                # This allows loggers to have multiple handlers of the same type (e.g., multiple
                # FileHandlers writing to different files), while preventing duplicate handler objects
                if handler not in logger.handlers:
                    logger.addHandler(handler)


def get_log_files() -> list[str]:
    """Get the log files from all loggers."""
    log_files = []
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        log_files.extend(
            [handler.baseFilename for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
        )
    return list(set(log_files))
