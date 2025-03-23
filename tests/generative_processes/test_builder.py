import pytest

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
)


def test_build_hidden_markov_model():
    hmm = build_hidden_markov_model("even_ones", p=0.5)
    assert hmm.vocab_size == 2

    with pytest.raises(ValueError):  # noqa: PT011
        build_hidden_markov_model("fanizza", alpha=2000, lamb=0.49)


def test_build_generalized_hidden_markov_model():
    ghmm = build_generalized_hidden_markov_model("even_ones", p=0.5)
    assert ghmm.vocab_size == 2

    ghmm = build_generalized_hidden_markov_model("fanizza", alpha=2000, lamb=0.49)
    assert ghmm.vocab_size == 2

    with pytest.raises(ValueError):  # noqa: PT011
        build_generalized_hidden_markov_model("dummy")
