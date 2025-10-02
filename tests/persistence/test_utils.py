from pathlib import Path

import pytest

from simplexity.persistence.utils import format_step_number, get_checkpoint_path, parse_checkpoint_step


class TestParseCheckpointStep:
    """Test parse_checkpoint_step function."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("12345/model.pt", 12345),
            ("checkpoints/12345/model.pt", 12345),
            ("path/to/500/model.pt", 500),
            ("0/model.pt", 0),
            ("prefix/run_name/12345/model.pt", 12345),
        ],
    )
    def test_directory_model_format(self, path: str, expected: int):
        """Test parsing {step}/model.pt format."""
        assert parse_checkpoint_step(path) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "model.pt",
            "checkpoint.pt",
            "weights/model.eqx",
            "random_file.txt",
            "nonumeric/model.pt",
        ],
    )
    def test_no_match_returns_none(self, path: str):
        """Test paths that should not match any pattern."""
        assert parse_checkpoint_step(path) is None

    def test_zero_padded_step_numbers(self):
        """Test that zero-padded step numbers are correctly parsed."""
        assert parse_checkpoint_step("0000/model.pt") == 0


class TestGetCheckpointPath:
    """Test get_checkpoint_path function."""

    def test_basic_path_construction(self):
        """Test basic checkpoint path construction."""
        path = get_checkpoint_path(Path("checkpoints"), 12345)
        assert path == Path("checkpoints/12345/model.pt")

    def test_custom_filename(self):
        """Test with custom filename."""
        path = get_checkpoint_path(Path("weights"), 100, "state.pt")
        assert path == Path("weights/100/state.pt")

    @pytest.mark.parametrize(
        ("directory", "step", "filename", "expected"),
        [
            (Path("checkpoints"), 0, "model.pt", Path("checkpoints/0/model.pt")),
            (Path("runs/exp1"), 1000, "checkpoint.pt", Path("runs/exp1/1000/checkpoint.pt")),
            (Path("."), 42, "model.pt", Path("42/model.pt")),
        ],
    )
    def test_parametrized_paths(self, directory: Path, step: int, filename: str, expected: Path):
        """Test various path combinations."""
        assert get_checkpoint_path(directory, step, filename) == expected


class TestFormatStepNumber:
    """Test format_step_number function."""

    def test_basic_formatting(self):
        """Test basic zero-padding behavior."""
        assert format_step_number(42, max_steps=100) == "042"
        assert format_step_number(5, max_steps=1000) == "0005"

    def test_no_padding_needed(self):
        """Test when step already has maximum width."""
        assert format_step_number(999, max_steps=999) == "999"
        assert format_step_number(100, max_steps=100) == "100"

    def test_zero_step(self):
        """Test formatting step 0."""
        assert format_step_number(0, max_steps=100) == "000"
        assert format_step_number(0, max_steps=10000) == "00000"

    @pytest.mark.parametrize(
        ("step", "max_steps", "expected"),
        [
            (0, 999, "000"),
            (1, 999, "001"),
            (42, 999, "042"),
            (999, 999, "999"),
            (0, 100000, "000000"),
            (42, 100000, "000042"),
            (12345, 100000, "012345"),
            (100000, 100000, "100000"),
        ],
    )
    def test_parametrized_formatting(self, step: int, max_steps: int, expected: str):
        """Test various step and max_steps combinations."""
        assert format_step_number(step, max_steps) == expected

    def test_lexicographic_ordering(self):
        """Verify that formatted strings sort lexicographically."""
        max_steps = 10000
        formatted = [format_step_number(i, max_steps) for i in [1, 10, 100, 1000, 9999]]
        assert formatted == sorted(formatted)

    def test_width_computation(self):
        """Verify format_step_number computes width correctly."""
        max_steps = 100000
        step = 42
        formatted = format_step_number(step, max_steps)
        expected_width = len(str(max_steps))
        assert len(formatted) == expected_width
