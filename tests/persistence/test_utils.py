import pytest

from simplexity.persistence.utils import compute_step_width, format_step_number, parse_checkpoint_step


class TestParseCheckpointStep:
    """Test parse_checkpoint_step function."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("model_weights/step_12345.pt", 12345),
            ("step_12345.pt", 12345),
            ("step_00012345.pt", 12345),
            ("checkpoints/step_500.pt", 500),
            ("path/to/step_999.pt", 999),
        ],
    )
    def test_step_underscore_format(self, path: str, expected: int):
        """Test parsing step_XXXX.pt format."""
        assert parse_checkpoint_step(path) == expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("step-12345.pt", 12345),
            ("step-00500.pt", 500),
            ("model_weights/step-999.pt", 999),
        ],
    )
    def test_step_hyphen_format(self, path: str, expected: int):
        """Test parsing step-XXXX.pt format."""
        assert parse_checkpoint_step(path) == expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("12345/model.pt", 12345),
            ("checkpoints/12345/model.pt", 12345),
            ("path/to/500/model.pt", 500),
            ("0/model.pt", 0),
        ],
    )
    def test_directory_model_format(self, path: str, expected: int):
        """Test parsing XXXX/model.pt format."""
        assert parse_checkpoint_step(path) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "model.pt",
            "checkpoint.pt",
            "step.pt",
            "weights/model.eqx",
            "random_file.txt",
            "step_abc.pt",
            "nonumeric/model.pt",
        ],
    )
    def test_no_match_returns_none(self, path: str):
        """Test paths that should not match any pattern."""
        assert parse_checkpoint_step(path) is None

    def test_zero_padded_step_numbers(self):
        """Test that zero-padded step numbers are correctly parsed."""
        assert parse_checkpoint_step("step_00042.pt") == 42
        assert parse_checkpoint_step("step_00000.pt") == 0
        assert parse_checkpoint_step("0000/model.pt") == 0

    def test_step_pattern_takes_precedence_over_directory(self):
        """Test that step_*.pt pattern takes precedence over directory pattern."""
        assert parse_checkpoint_step("path/step_200.pt") == 200
        assert parse_checkpoint_step("checkpoints/step_999.pt") == 999

    def test_windows_paths(self):
        """Test Windows-style paths with backslashes."""
        path_unix = "checkpoints/12345/model.pt"
        assert parse_checkpoint_step(path_unix) == 12345

    def test_s3_style_keys(self):
        """Test S3 object key formats."""
        assert parse_checkpoint_step("s3://bucket/prefix/step_12345.pt") == 12345
        assert parse_checkpoint_step("prefix/run_name/12345/model.pt") == 12345


class TestComputeStepWidth:
    """Test compute_step_width function."""

    def test_single_digit(self):
        """Test with max_steps requiring 1 digit."""
        assert compute_step_width(9) == 1
        assert compute_step_width(1) == 1

    def test_two_digits(self):
        """Test with max_steps requiring 2 digits."""
        assert compute_step_width(10) == 2
        assert compute_step_width(99) == 2

    def test_three_digits(self):
        """Test with max_steps requiring 3 digits."""
        assert compute_step_width(100) == 3
        assert compute_step_width(999) == 3

    @pytest.mark.parametrize(
        ("max_steps", "expected_width"),
        [
            (1, 1),
            (9, 1),
            (10, 2),
            (99, 2),
            (100, 3),
            (999, 3),
            (1000, 4),
            (9999, 4),
            (10000, 5),
            (99999, 5),
            (100000, 6),
        ],
    )
    def test_parametrized_widths(self, max_steps: int, expected_width: int):
        """Test various max_steps values."""
        assert compute_step_width(max_steps) == expected_width

    def test_large_step_counts(self):
        """Test with very large step counts."""
        assert compute_step_width(1000000) == 7
        assert compute_step_width(10000000) == 8


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

    def test_consistency_with_compute_step_width(self):
        """Verify format_step_number uses compute_step_width correctly."""
        max_steps = 100000
        step = 42
        formatted = format_step_number(step, max_steps)
        expected_width = compute_step_width(max_steps)
        assert len(formatted) == expected_width
