import pytest

from simplexity.utils.config_resolution import (
    compute_generator_sequence_length,
    compute_model_context_length,
    compute_model_vocab_size,
)


class TestComputeGeneratorSequenceLength:
    """Test compute_generator_sequence_length function."""

    @pytest.mark.parametrize(
        ("use_bos", "use_eos", "expected"),
        [
            (False, False, 65),
            (True, False, 64),
            (False, True, 64),
            (True, True, 63),
        ],
    )
    def test_bos_eos_combinations(self, use_bos: bool, use_eos: bool, expected: int):
        """Test all combinations of BOS and EOS tokens with model_n_ctx=64."""
        assert compute_generator_sequence_length(64, use_bos=use_bos, use_eos=use_eos) == expected

    def test_invalid_configuration_raises_error(self):
        """Test that invalid configurations raise ValueError."""
        with pytest.raises(ValueError, match="non-positive generator sequence length"):
            compute_generator_sequence_length(model_n_ctx=1, use_bos=True, use_eos=True)
        with pytest.raises(AssertionError, match="must be positive"):
            compute_generator_sequence_length(model_n_ctx=0, use_bos=True, use_eos=False)


class TestComputeModelContextLength:
    """Test compute_model_context_length function."""

    @pytest.mark.parametrize(
        ("use_bos", "use_eos", "expected"),
        [
            (False, False, 63),
            (True, False, 64),
            (False, True, 64),
            (True, True, 65),
        ],
    )
    def test_bos_eos_combinations(self, use_bos: bool, use_eos: bool, expected: int):
        """Test all combinations of BOS and EOS tokens with generator_seq_len=64."""
        assert compute_model_context_length(64, use_bos=use_bos, use_eos=use_eos) == expected

    @pytest.mark.parametrize(
        ("use_bos", "use_eos", "expected"),
        [
            (False, False, 511),
            (True, False, 512),
            (False, True, 512),
            (True, True, 513),
        ],
    )
    def test_parametrized_cases(self, use_bos: bool, use_eos: bool, expected: int):
        """Test all combinations with generator_seq_len=512."""
        assert compute_model_context_length(512, use_bos=use_bos, use_eos=use_eos) == expected

    def test_invalid_inputs_raise_error(self):
        """Test that invalid inputs raise AssertionError."""
        with pytest.raises(AssertionError, match="must be positive"):
            compute_model_context_length(generator_seq_len=0, use_bos=True, use_eos=True)

    @pytest.mark.parametrize(
        ("use_bos", "use_eos"),
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_inverse_relationship(self, use_bos: bool, use_eos: bool):
        """Verify inverse relationship with compute_generator_sequence_length."""
        model_n_ctx = 512
        gen_seq_len = compute_generator_sequence_length(model_n_ctx, use_bos=use_bos, use_eos=use_eos)
        recovered_n_ctx = compute_model_context_length(gen_seq_len, use_bos=use_bos, use_eos=use_eos)
        assert recovered_n_ctx == model_n_ctx

    @pytest.mark.parametrize("model_n_ctx", [1, 64, 128, 512, 1024])
    @pytest.mark.parametrize("use_bos", [True, False])
    @pytest.mark.parametrize("use_eos", [True, False])
    def test_round_trip_consistency(self, model_n_ctx: int, use_bos: bool, use_eos: bool):
        """Verify round-trip conversion maintains original value."""
        if model_n_ctx == 1 and use_bos and use_eos:
            pytest.skip("Configuration would produce invalid sequence length")
        gen_seq_len = compute_generator_sequence_length(model_n_ctx, use_bos=use_bos, use_eos=use_eos)
        recovered = compute_model_context_length(gen_seq_len, use_bos=use_bos, use_eos=use_eos)
        assert recovered == model_n_ctx


class TestComputeModelVocabSize:
    """Test compute_model_vocab_size function."""

    @pytest.mark.parametrize(
        ("use_bos", "use_eos", "expected"),
        [
            (False, False, 100),
            (True, False, 101),
            (False, True, 101),
            (True, True, 102),
        ],
    )
    def test_bos_eos_combinations(self, use_bos: bool, use_eos: bool, expected: int):
        """Test all combinations of BOS and EOS tokens with generator_vocab_size=100."""
        assert compute_model_vocab_size(100, use_bos=use_bos, use_eos=use_eos) == expected

    @pytest.mark.parametrize(
        ("generator_vocab_size", "use_bos", "use_eos", "expected"),
        [
            (50257, False, False, 50257),
            (50257, True, False, 50258),
            (50257, True, True, 50259),
        ],
    )
    def test_parametrized_cases(self, generator_vocab_size: int, use_bos: bool, use_eos: bool, expected: int):
        """Test various combinations of vocab size and special tokens."""
        assert compute_model_vocab_size(generator_vocab_size, use_bos=use_bos, use_eos=use_eos) == expected

    def test_invalid_vocab_size_raises_error(self):
        """Test that non-positive vocab sizes raise AssertionError."""
        with pytest.raises(AssertionError, match="must be positive"):
            compute_model_vocab_size(generator_vocab_size=0, use_bos=True, use_eos=False)
