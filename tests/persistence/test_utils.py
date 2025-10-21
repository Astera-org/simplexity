from pathlib import Path

import pytest

from simplexity.persistence.utils import format_step_number, get_checkpoint_path, parse_checkpoint_step


class TestParseCheckpointStep:
    """Test parse_checkpoint_step function."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("12345/model.pt", 12345),
            ("checkpoints/12345/checkpoint.pt", 12345),
            ("path/to/500/state.pt", 500),
            ("0000/weights.eqx", 0),
            ("prefix/run_name/12345/model.pkl", 12345),
        ],
    )
    def test_directory_model_format(self, path: str, expected: int):
        """Test parsing {step}/filename format with various filenames and zero-padding."""
        assert parse_checkpoint_step(path) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "model.pt",
            "weights/model.eqx",
            "abc123/model.pt",
            "123abc/checkpoint.pt",
        ],
    )
    def test_no_match_returns_none(self, path: str):
        """Test paths with numbers in invalid positions return None."""
        assert parse_checkpoint_step(path) is None


class TestGetCheckpointPath:
    """Test get_checkpoint_path function."""

    @pytest.mark.parametrize(
        ("directory", "step", "filename", "max_steps", "expected"),
        [
            (Path("checkpoints"), 12345, "model.pt", None, Path("checkpoints/12345/model.pt")),
            (Path("runs/exp1"), 1000, "checkpoint.pt", None, Path("runs/exp1/1000/checkpoint.pt")),
            (Path("weights"), 42, "state.eqx", 100000, Path("weights/000042/state.eqx")),
            (Path("checkpoints"), 0, "model.pt", 999, Path("checkpoints/000/model.pt")),
        ],
    )
    def test_parametrized_paths(self, directory: Path, step: int, filename: str, max_steps: int | None, expected: Path):
        """Test various path combinations including custom filenames and zero-padding."""
        assert get_checkpoint_path(directory, step, filename, max_steps) == expected

    def test_negative_step_raises_error(self):
        """Test that negative step values raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            get_checkpoint_path(Path("checkpoints"), -1)

    def test_invalid_filename_raises_error(self):
        """Test that invalid filenames raise ValueError."""
        with pytest.raises(ValueError, match="must have one of these extensions"):
            get_checkpoint_path(Path("checkpoints"), 100, "invalid.txt")


class TestFormatStepNumber:
    """Test format_step_number function."""

    @pytest.mark.parametrize(
        ("step", "max_steps", "expected"),
        [
            (0, 999, "000"),
            (42, 999, "042"),
            (999, 999, "999"),
            (42, 100000, "000042"),
            (100000, 100000, "100000"),
        ],
    )
    def test_parametrized_formatting(self, step: int, max_steps: int, expected: str):
        """Test various step and max_steps combinations with zero-padding."""
        assert format_step_number(step, max_steps) == expected

    def test_lexicographic_ordering(self):
        """Verify that formatted strings sort lexicographically."""
        max_steps = 10000
        formatted = [format_step_number(i, max_steps) for i in [1, 10, 100, 1000, 9999]]
        assert formatted == sorted(formatted)

    def test_invalid_step_raises_error(self):
        """Test that invalid step values raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and"):
            format_step_number(-1, max_steps=100)
        with pytest.raises(ValueError, match="must be between 0 and"):
            format_step_number(101, max_steps=100)
