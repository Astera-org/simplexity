"""Test automatic vocab size and special token computation."""

from omegaconf import DictConfig, OmegaConf

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.run import compute_vocab_and_special_tokens


def test_compute_vocab_no_special_tokens():
    """Test vocab computation with no special tokens."""
    generator = build_hidden_markov_model("mess3", x=0.15, a=0.6)
    cfg = OmegaConf.create({"use_bos": False, "use_eos": False})

    compute_vocab_and_special_tokens(cfg, generator)

    assert cfg.bos_token is None
    assert cfg.eos_token is None
    assert cfg.vocab_size == 3


def test_compute_vocab_with_bos():
    """Test vocab computation with BOS token."""
    generator = build_hidden_markov_model("mess3", x=0.15, a=0.6)
    cfg = OmegaConf.create({"use_bos": True, "use_eos": False})

    compute_vocab_and_special_tokens(cfg, generator)

    assert cfg.bos_token == 3
    assert cfg.eos_token is None
    assert cfg.vocab_size == 4


def test_compute_vocab_with_eos():
    """Test vocab computation with EOS token."""
    generator = build_hidden_markov_model("zero_one_random", p=0.5)
    cfg = OmegaConf.create({"use_bos": False, "use_eos": True})

    compute_vocab_and_special_tokens(cfg, generator)

    assert cfg.bos_token is None
    assert cfg.eos_token == 2
    assert cfg.vocab_size == 3


def test_compute_vocab_with_both_special_tokens():
    """Test vocab computation with both BOS and EOS tokens."""
    generator = build_hidden_markov_model("zero_one_random", p=0.5)
    cfg = OmegaConf.create({"use_bos": True, "use_eos": True})

    compute_vocab_and_special_tokens(cfg, generator)

    assert cfg.bos_token == 2
    assert cfg.eos_token == 3
    assert cfg.vocab_size == 4


def test_compute_vocab_different_base_sizes():
    """Test vocab computation works correctly with different base vocab sizes."""
    generator_small = build_hidden_markov_model("zero_one_random", p=0.5)
    generator_large = build_hidden_markov_model("days_of_week")

    cfg_small = OmegaConf.create({"use_bos": True, "use_eos": False})
    cfg_large = OmegaConf.create({"use_bos": True, "use_eos": False})

    compute_vocab_and_special_tokens(cfg_small, generator_small)
    compute_vocab_and_special_tokens(cfg_large, generator_large)

    assert cfg_small.bos_token == 2
    assert cfg_small.vocab_size == 3

    assert cfg_large.bos_token == 11
    assert cfg_large.vocab_size == 12
