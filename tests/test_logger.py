"""Tests for the simplexity logger module."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import contextlib
import functools
import logging
import logging.config
from pathlib import Path

from simplexity.logger import SIMPLEXITY_LOGGER, get_log_files


@contextlib.contextmanager
def preserve_logging_state():
    """Context manager that preserves and restores the global logging configuration."""
    # Save current logging state
    saved_logger_dict = logging.root.manager.loggerDict.copy()
    saved_root_handlers = logging.root.handlers.copy()
    saved_root_level = logging.root.level

    try:
        yield
    finally:
        # Restore logging state
        logging.shutdown()
        # Clear current loggerDict and restore saved one
        logging.root.manager.loggerDict.clear()
        logging.root.manager.loggerDict.update(saved_logger_dict)
        # Restore root handlers and level
        logging.root.handlers = saved_root_handlers
        logging.root.level = saved_root_level


def with_preserved_logging_state(func):
    """Decorator that preserves and restores the global logging configuration for a test function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with preserve_logging_state():
            return func(*args, **kwargs)

    return wrapper


def test_simplexity_logger() -> None:
    """Test that the logger is created with the correct name."""
    assert SIMPLEXITY_LOGGER.name == "simplexity"
    assert isinstance(SIMPLEXITY_LOGGER, logging.Logger)


@with_preserved_logging_state
def test_get_log_files_no_files() -> None:
    """Test that the log files are returned correctly."""
    assert not get_log_files()

    logging.config.dictConfig(
        {
            "version": 1,
            "handlers": {
                "stream": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "root": {
                    "handlers": ["stream"],
                },
                "simplexity": {
                    "handlers": ["stream"],
                },
            },
        }
    )
    assert not get_log_files()


@with_preserved_logging_state
def test_get_log_files_with_files(tmp_path: Path) -> None:
    """Test that the log files are returned correctly."""
    test_1_log_file = str(tmp_path / "test_1.log")
    test_2_log_file = str(tmp_path / "test_2.log")
    test_3_log_file = str(tmp_path / "test_3.log")

    logging.config.dictConfig(
        {
            "version": 1,
            "handlers": {
                "stream": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file_1": {
                    "class": "logging.FileHandler",
                    "filename": test_1_log_file,
                },
                "file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_2_log_file,
                },
                "file_3": {
                    "class": "logging.FileHandler",
                    "filename": test_3_log_file,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream", "file_2"],
                },
                "simplexity": {
                    "handlers": ["file_1", "file_3"],
                },
                "other": {
                    "handlers": ["file_1"],
                },
            },
        }
    )

    log_files = get_log_files()
    assert len(log_files) == 3
    assert set(log_files) == {test_1_log_file, test_2_log_file, test_3_log_file}
