"""Managed run demo."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

from hydra import compose, initialize_config_dir

from tests.end_to_end.training import train

_E2E_DIR = Path(__file__).parent


def test_training() -> None:
    """Test training."""
    with initialize_config_dir(str(_E2E_DIR / "configs"), version_base="1.2"):
        cfg = compose(config_name="config.yaml")
    train(cfg)
