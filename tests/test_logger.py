"""Tests for the simplexity logger module."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import logging

from simplexity.logger import SIMPLEXITY_LOGGER


def test_simplexity_logger() -> None:
    """Test that the logger is created with the correct name."""
    assert SIMPLEXITY_LOGGER.name == "simplexity"
    assert isinstance(SIMPLEXITY_LOGGER, logging.Logger)
