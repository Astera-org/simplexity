def compute_generator_sequence_length(model_n_ctx: int, use_bos: bool, use_eos: bool = False) -> int:
    """Compute the generator's sequence length from model context length and special token usage.

    The relationship is: model_n_ctx = generator_seq_len - 1 + BOS + EOS

    Solving for generator_seq_len: generator_seq_len = model_n_ctx + 1 - BOS - EOS

    Args:
        model_n_ctx: The model's context length (number of input positions it processes)
        use_bos: Whether a beginning-of-sequence token is prepended during data generation
        use_eos: Whether an end-of-sequence token is appended during data generation

    Returns:
        The sequence length to configure for the data generator

    Raises:
        ValueError: If the resulting generator sequence length would be non-positive

    Examples:
        >>> compute_generator_sequence_length(model_n_ctx=512, use_bos=True, use_eos=False)
        512
        >>> compute_generator_sequence_length(model_n_ctx=512, use_bos=False, use_eos=False)
        513
        >>> compute_generator_sequence_length(model_n_ctx=512, use_bos=True, use_eos=True)
        511
    """
    assert model_n_ctx > 0, f"model_n_ctx must be positive, got {model_n_ctx}"

    result = model_n_ctx + 1 - int(use_bos) - int(use_eos)
    if result <= 0:
        raise ValueError(
            f"Invalid configuration: model_n_ctx={model_n_ctx}, use_bos={use_bos}, use_eos={use_eos} "
            f"results in non-positive generator sequence length ({result})"
        )
    return result


def compute_model_context_length(generator_seq_len: int, use_bos: bool, use_eos: bool = False) -> int:
    """Compute the model's context length from generator sequence length and special token usage.

    The relationship is: model_n_ctx = generator_seq_len - 1 + BOS + EOS

    Args:
        generator_seq_len: The sequence length configured for the data generator
        use_bos: Whether a beginning-of-sequence token is prepended during data generation
        use_eos: Whether an end-of-sequence token is appended during data generation

    Returns:
        The context length for the model (number of input positions it will process)

    Raises:
        ValueError: If the resulting model context length would be non-positive

    Examples:
        >>> compute_model_context_length(generator_seq_len=512, use_bos=True, use_eos=False)
        512
        >>> compute_model_context_length(generator_seq_len=513, use_bos=False, use_eos=False)
        512
        >>> compute_model_context_length(generator_seq_len=511, use_bos=True, use_eos=True)
        512
    """
    assert generator_seq_len > 0, f"generator_seq_len must be positive, got {generator_seq_len}"

    result = generator_seq_len - 1 + int(use_bos) + int(use_eos)
    if result <= 0:
        raise ValueError(
            f"Invalid configuration: generator_seq_len={generator_seq_len}, use_bos={use_bos}, use_eos={use_eos} "
            f"results in non-positive model context length ({result})"
        )
    return result


def compute_model_vocab_size(generator_vocab_size: int, use_bos: bool, use_eos: bool = False) -> int:
    """Compute the model's vocabulary size from generator vocab and special tokens.

    When BOS or EOS tokens are used during data generation, they are added to the vocabulary,
    increasing the total vocab size the model needs to handle.

    Args:
        generator_vocab_size: The vocabulary size of the data generator
        use_bos: Whether a beginning-of-sequence token is used during data generation
        use_eos: Whether an end-of-sequence token is used during data generation

    Returns:
        The vocabulary size the model should be configured with

    Raises:
        ValueError: If generator_vocab_size is non-positive

    Examples:
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=False)
        101
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=True)
        102
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=False, use_eos=False)
        100
    """
    assert generator_vocab_size > 0, f"generator_vocab_size must be positive, got {generator_vocab_size}"
    return generator_vocab_size + int(use_bos) + int(use_eos)
