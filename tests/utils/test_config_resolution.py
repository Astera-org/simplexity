import pytest

from simplexity.utils.config_resolution import (
    compute_generator_sequence_length,
    compute_model_context_length,
    compute_model_vocab_size,
)


class TestComputeGeneratorSequenceLength:
    """Test compute_generator_sequence_length function."""

    def test_with_bos_token(self):
        """When BOS is used, generator_seq_len should equal model_n_ctx."""
        assert compute_generator_sequence_length(model_n_ctx=512, use_bos=True) == 512
        assert compute_generator_sequence_length(model_n_ctx=100, use_bos=True) == 100

    def test_without_bos_token(self):
        """When BOS is not used, generator_seq_len should be model_n_ctx + 1."""
        assert compute_generator_sequence_length(model_n_ctx=512, use_bos=False) == 513
        assert compute_generator_sequence_length(model_n_ctx=100, use_bos=False) == 101

    @pytest.mark.parametrize(
        ("model_n_ctx", "use_bos", "expected"),
        [
            (1, True, 1),
            (1, False, 2),
            (64, True, 64),
            (64, False, 65),
            (1024, True, 1024),
            (1024, False, 1025),
        ],
    )
    def test_parametrized_cases(self, model_n_ctx: int, use_bos: bool, expected: int):
        """Test various combinations of model_n_ctx and use_bos."""
        assert compute_generator_sequence_length(model_n_ctx, use_bos) == expected

    def test_zero_context_with_bos(self):
        """Edge case: zero context length with BOS."""
        assert compute_generator_sequence_length(model_n_ctx=0, use_bos=True) == 0

    def test_zero_context_without_bos(self):
        """Edge case: zero context length without BOS."""
        assert compute_generator_sequence_length(model_n_ctx=0, use_bos=False) == 1


class TestComputeModelContextLength:
    """Test compute_model_context_length function."""

    def test_with_bos_token(self):
        """When BOS is used, model_n_ctx should equal generator_seq_len."""
        assert compute_model_context_length(generator_seq_len=512, use_bos=True) == 512
        assert compute_model_context_length(generator_seq_len=100, use_bos=True) == 100

    def test_without_bos_token(self):
        """When BOS is not used, model_n_ctx should be generator_seq_len - 1."""
        assert compute_model_context_length(generator_seq_len=513, use_bos=False) == 512
        assert compute_model_context_length(generator_seq_len=101, use_bos=False) == 100

    @pytest.mark.parametrize(
        ("generator_seq_len", "use_bos", "expected"),
        [
            (1, True, 1),
            (2, False, 1),
            (64, True, 64),
            (65, False, 64),
            (1024, True, 1024),
            (1025, False, 1024),
        ],
    )
    def test_parametrized_cases(self, generator_seq_len: int, use_bos: bool, expected: int):
        """Test various combinations of generator_seq_len and use_bos."""
        assert compute_model_context_length(generator_seq_len, use_bos) == expected

    def test_inverse_relationship_with_bos(self):
        """Verify inverse relationship with compute_generator_sequence_length when using BOS."""
        model_n_ctx = 512
        use_bos = True
        gen_seq_len = compute_generator_sequence_length(model_n_ctx, use_bos)
        recovered_n_ctx = compute_model_context_length(gen_seq_len, use_bos)
        assert recovered_n_ctx == model_n_ctx

    def test_inverse_relationship_without_bos(self):
        """Verify inverse relationship with compute_generator_sequence_length without BOS."""
        model_n_ctx = 512
        use_bos = False
        gen_seq_len = compute_generator_sequence_length(model_n_ctx, use_bos)
        recovered_n_ctx = compute_model_context_length(gen_seq_len, use_bos)
        assert recovered_n_ctx == model_n_ctx

    @pytest.mark.parametrize("model_n_ctx", [1, 64, 128, 512, 1024])
    @pytest.mark.parametrize("use_bos", [True, False])
    def test_round_trip_consistency(self, model_n_ctx: int, use_bos: bool):
        """Verify round-trip conversion maintains original value."""
        gen_seq_len = compute_generator_sequence_length(model_n_ctx, use_bos)
        recovered = compute_model_context_length(gen_seq_len, use_bos)
        assert recovered == model_n_ctx


class TestComputeModelVocabSize:
    """Test compute_model_vocab_size function."""

    def test_no_special_tokens(self):
        """When no special tokens are used, vocab size should equal generator vocab."""
        assert compute_model_vocab_size(generator_vocab_size=100, use_bos=False, use_eos=False) == 100

    def test_with_bos_only(self):
        """When only BOS is used, vocab size should be generator_vocab + 1."""
        assert compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=False) == 101

    def test_with_eos_only(self):
        """When only EOS is used, vocab size should be generator_vocab + 1."""
        assert compute_model_vocab_size(generator_vocab_size=100, use_bos=False, use_eos=True) == 101

    def test_with_both_special_tokens(self):
        """When both BOS and EOS are used, vocab size should be generator_vocab + 2."""
        assert compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=True) == 102

    @pytest.mark.parametrize(
        ("generator_vocab_size", "use_bos", "use_eos", "expected"),
        [
            (100, False, False, 100),
            (100, True, False, 101),
            (100, False, True, 101),
            (100, True, True, 102),
            (1, False, False, 1),
            (1, True, True, 3),
            (50257, False, False, 50257),
            (50257, True, False, 50258),
            (50257, True, True, 50259),
        ],
    )
    def test_parametrized_cases(self, generator_vocab_size: int, use_bos: bool, use_eos: bool, expected: int):
        """Test various combinations of vocab size and special tokens."""
        assert compute_model_vocab_size(generator_vocab_size, use_bos, use_eos) == expected

    def test_minimal_vocab_with_tokens(self):
        """Edge case: minimal vocabulary with special tokens."""
        assert compute_model_vocab_size(generator_vocab_size=2, use_bos=True, use_eos=True) == 4

    def test_large_vocab(self):
        """Test with large vocabulary sizes."""
        assert compute_model_vocab_size(generator_vocab_size=100000, use_bos=True, use_eos=True) == 100002
