def compute_generator_sequence_length(model_n_ctx: int, use_bos: bool) -> int:
    """Compute the generator's sequence length from model context length and BOS usage.

    The relationship is: model_n_ctx = generator_seq_len - 1 + BOS

    Solving for generator_seq_len: generator_seq_len = model_n_ctx + 1 - BOS

    Args:
        model_n_ctx: The model's context length (number of input positions it processes)
        use_bos: Whether a beginning-of-sequence token is prepended during data generation

    Returns:
        The sequence length to configure for the data generator

    Examples:
        >>> compute_generator_sequence_length(model_n_ctx=512, use_bos=True)
        512
        >>> compute_generator_sequence_length(model_n_ctx=512, use_bos=False)
        513
    """
    return model_n_ctx + 1 - int(use_bos)


def compute_model_context_length(generator_seq_len: int, use_bos: bool) -> int:
    """Compute the model's context length from generator sequence length and BOS usage.

    The relationship is: model_n_ctx = generator_seq_len - 1 + BOS

    Args:
        generator_seq_len: The sequence length configured for the data generator
        use_bos: Whether a beginning-of-sequence token is prepended during data generation

    Returns:
        The context length for the model (number of input positions it will process)

    Examples:
        >>> compute_model_context_length(generator_seq_len=512, use_bos=True)
        512
        >>> compute_model_context_length(generator_seq_len=513, use_bos=False)
        512
    """
    return generator_seq_len - 1 + int(use_bos)


def compute_model_vocab_size(generator_vocab_size: int, use_bos: bool, use_eos: bool) -> int:
    """Compute the model's vocabulary size from generator vocab and special tokens.

    When BOS or EOS tokens are used during data generation, they are added to the vocabulary,
    increasing the total vocab size the model needs to handle.

    Args:
        generator_vocab_size: The vocabulary size of the data generator
        use_bos: Whether a beginning-of-sequence token is used during data generation
        use_eos: Whether an end-of-sequence token is used during data generation

    Returns:
        The vocabulary size the model should be configured with

    Examples:
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=False)
        101
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=True, use_eos=True)
        102
        >>> compute_model_vocab_size(generator_vocab_size=100, use_bos=False, use_eos=False)
        100
    """
    return generator_vocab_size + int(use_bos) + int(use_eos)
