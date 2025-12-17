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

from simplexity.logger import SIMPLEXITY_LOGGER, get_log_files, remove_file_handlers, remove_log_file, remove_log_files


def clear_logging_state():
    """Clear the logging state."""
    logging.shutdown()
    logging.root.manager.loggerDict.clear()
    logging.root.handlers.clear()
    logging.root.level = logging.NOTSET
    logging.root.propagate = True
    logging.root.disabled = False


@contextlib.contextmanager
def clean_slate():
    """Context manager that saves all logging state, clears it during yield, then restores it."""
    # Save all existing logging state
    saved_logger_dict = logging.root.manager.loggerDict.copy()
    saved_logger_states = {}
    for logger_name, logger in saved_logger_dict.items():
        if isinstance(logger, logging.Logger):
            saved_logger_states[logger_name] = {
                "handlers": logger.handlers.copy(),
                "level": logger.level,
                "propagate": logger.propagate,
                "disabled": logger.disabled,
            }

    # Save root logger state
    saved_root_handlers = logging.root.handlers.copy()
    saved_root_level = logging.root.level
    saved_root_propagate = logging.root.propagate
    saved_root_disabled = logging.root.disabled

    try:
        clear_logging_state()
        yield
    finally:
        clear_logging_state()

        # Restore loggerDict with original logger objects
        logging.root.manager.loggerDict.update(saved_logger_dict)

        # Restore each logger's state (handlers, level, etc.)
        for logger_name, logger_state in saved_logger_states.items():
            logger = saved_logger_dict[logger_name]
            if isinstance(logger, logging.Logger):
                logger.handlers = logger_state["handlers"]
                logger.level = logger_state["level"]
                logger.propagate = logger_state["propagate"]
                logger.disabled = logger_state["disabled"]

        # Restore root logger state
        logging.root.handlers = saved_root_handlers
        logging.root.level = saved_root_level
        logging.root.propagate = saved_root_propagate
        logging.root.disabled = saved_root_disabled


def with_clean_slate(func):
    """Decorator that preserves and restores the global logging configuration for a test function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with clean_slate():
            return func(*args, **kwargs)

    return wrapper


def test_simplexity_logger() -> None:
    """Test that the logger is created with the correct name."""
    assert SIMPLEXITY_LOGGER.name == "simplexity"
    assert isinstance(SIMPLEXITY_LOGGER, logging.Logger)


@with_clean_slate
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


@with_clean_slate
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


@with_clean_slate
def test_remove_file_handlers_with_no_files(tmp_path: Path) -> None:
    """Test that the file handlers are removed correctly."""
    logging.config.dictConfig(
        {
            "version": 1,
            "handlers": {
                "stream": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(tmp_path / "test.log"),
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream"],
                },
                "simplexity": {
                    "handlers": ["file"],
                },
            },
        }
    )
    # no file handlers to remove
    root_logger = logging.getLogger("root")
    assert len(root_logger.handlers) == 1
    remove_file_handlers(root_logger)
    assert len(root_logger.handlers) == 1

    # no matching file handler to remove
    simplexity_logger = logging.getLogger("simplexity")
    assert len(simplexity_logger.handlers) == 1
    remove_file_handlers(simplexity_logger, str(tmp_path / "different.log"))
    assert len(simplexity_logger.handlers) == 1


@with_clean_slate
def test_remove_file_handlers(tmp_path: Path) -> None:
    """Test that the file handlers are removed correctly."""
    test_log_file_1 = str(tmp_path / "test_1.log")
    test_log_file_2 = str(tmp_path / "test_2.log")
    test_log_file_3 = str(tmp_path / "test_3.log")
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
                    "filename": test_log_file_1,
                },
                "file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "file_3": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_3,
                },
                "another_file_3": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_3,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream", "file_2"],
                },
                "simplexity": {
                    "handlers": ["stream", "file_1", "file_2"],
                },
                "other": {
                    "handlers": ["file_2", "file_3", "another_file_3"],
                },
            },
        }
    )
    # remove matching file handler
    root_logger = logging.getLogger("root")
    assert len(root_logger.handlers) == 2
    remove_file_handlers(root_logger, test_log_file_2)
    assert len(root_logger.handlers) == 1  # stream handler is still present

    # remove all file handlers
    simplexity_logger = logging.getLogger("simplexity")
    assert len(simplexity_logger.handlers) == 3
    remove_file_handlers(simplexity_logger)
    assert len(simplexity_logger.handlers) == 1  # stream handler is still present
    assert not any(isinstance(h, logging.FileHandler) for h in simplexity_logger.handlers)

    # remove multiple matching file handlers
    other_logger = logging.getLogger("other")
    assert len(other_logger.handlers) == 3
    remove_file_handlers(other_logger, test_log_file_3)
    assert len(other_logger.handlers) == 1  # file_2 handler is still present
    file_handler = other_logger.handlers[0]
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.baseFilename != test_log_file_3


@with_clean_slate
def test_remove_log_file(tmp_path: Path) -> None:
    """Test that the log file is removed correctly."""
    test_path_1 = tmp_path / "test_1.log"
    test_path_2 = tmp_path / "test_2.log"

    test_path_1.touch()
    test_path_2.touch()

    test_log_file_1 = str(test_path_1)
    test_log_file_2 = str(test_path_2)
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
                    "filename": test_log_file_1,
                },
                "file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "another_file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream", "file_1"],
                },
                "simplexity": {
                    "handlers": ["file_1", "file_2"],
                },
                "other": {
                    "handlers": ["file_2", "another_file_2"],
                },
            },
        }
    )

    root_logger = logging.getLogger("root")
    simplexity_logger = logging.getLogger("simplexity")
    other_logger = logging.getLogger("other")

    assert len(root_logger.handlers) == 2
    assert len(simplexity_logger.handlers) == 2
    assert len(other_logger.handlers) == 2
    assert test_path_1.exists()
    assert test_path_2.exists()

    remove_log_file(test_log_file_2)

    assert len(root_logger.handlers) == 2  # no file_2 handlers to remove
    assert len(simplexity_logger.handlers) == 1  # file_2 handler is removed
    assert len(other_logger.handlers) == 0  # both file_2 handlers are removed
    assert test_path_1.exists()
    assert not test_path_2.exists()


@with_clean_slate
def test_remove_log_files(tmp_path: Path) -> None:
    """Test that the log files are removed correctly."""
    test_path_1 = tmp_path / "test_1.log"
    test_path_2 = tmp_path / "test_2.log"
    test_path_3 = tmp_path / "test_3.log"

    test_path_1.touch()
    test_path_2.touch()
    test_path_3.touch()

    test_log_file_1 = str(test_path_1)
    test_log_file_2 = str(test_path_2)
    test_log_file_3 = str(test_path_3)
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
                    "filename": test_log_file_1,
                },
                "file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "another_file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "file_3": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_3,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream", "file_1"],
                },
                "simplexity": {
                    "handlers": ["file_1", "file_2"],
                },
                "other": {
                    "handlers": ["file_2", "another_file_2", "file_3"],
                },
            },
        }
    )

    root_logger = logging.getLogger("root")
    simplexity_logger = logging.getLogger("simplexity")
    other_logger = logging.getLogger("other")

    assert len(root_logger.handlers) == 2
    assert len(simplexity_logger.handlers) == 2
    assert len(other_logger.handlers) == 3
    assert test_path_1.exists()
    assert test_path_2.exists()

    remove_log_files()

    assert len(root_logger.handlers) == 1  # stream handler is still present
    assert len(simplexity_logger.handlers) == 0  # all file handlers are removed
    assert len(other_logger.handlers) == 0  # all file handlers are removed
    assert not test_path_1.exists()
    assert not test_path_2.exists()
    assert not test_path_3.exists()


@with_clean_slate
def test_remove_log_files_with_specific_files(tmp_path: Path) -> None:
    """Test that the log files are removed correctly."""
    test_path_1 = tmp_path / "test_1.log"
    test_path_2 = tmp_path / "test_2.log"
    test_path_3 = tmp_path / "test_3.log"

    test_path_1.touch()
    test_path_2.touch()
    test_path_3.touch()

    test_log_file_1 = str(test_path_1)
    test_log_file_2 = str(test_path_2)
    test_log_file_3 = str(test_path_3)
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
                    "filename": test_log_file_1,
                },
                "file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "another_file_2": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_2,
                },
                "file_3": {
                    "class": "logging.FileHandler",
                    "filename": test_log_file_3,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["stream", "file_1", "file_3"],
                },
                "simplexity": {
                    "handlers": ["file_1", "file_2"],
                },
                "other": {
                    "handlers": ["file_2", "another_file_2", "file_3"],
                },
            },
        }
    )

    root_logger = logging.getLogger("root")
    simplexity_logger = logging.getLogger("simplexity")
    other_logger = logging.getLogger("other")

    assert len(root_logger.handlers) == 3
    assert len(simplexity_logger.handlers) == 2
    assert len(other_logger.handlers) == 3
    assert test_path_1.exists()
    assert test_path_2.exists()
    assert test_path_3.exists()

    remove_log_files({test_log_file_2, test_log_file_3})

    assert len(root_logger.handlers) == 2  # stream and file_1 handler are still present
    assert len(simplexity_logger.handlers) == 1  # file_1 handler is still present
    assert len(other_logger.handlers) == 0  # file_3 and both file_2 handlers are removed
    assert test_path_1.exists()
    assert not test_path_2.exists()
    assert not test_path_3.exists()
