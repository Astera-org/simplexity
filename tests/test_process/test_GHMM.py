import numpy as np
from simplexity.process.GHMM import TransitionMatrixGHMM

def test_ghmm_initialization():
    T = np.array([
        [[0.5, 0.5],
         [0.3, 0.7]],
        [[0.2, 0.8],
         [0.6, 0.4]]
    ])
    ghmm = TransitionMatrixGHMM(T)
    
    assert ghmm.vocab_len == 2
    assert ghmm.latent_dim == 2
    assert np.allclose(ghmm.transition_matrices, T)

def test_steady_state_vector():
    T = np.array([
        [[0.25, 0.25],
         [0.25, 0.25]],
        [[0.25, 0.25],
         [0.25, 0.25]]
    ])
    ghmm = TransitionMatrixGHMM(T)
    
    expected_ssv = np.array([0.5, 0.5])
    assert np.allclose(ghmm.steady_state_vector, expected_ssv)

def test_word_probability():
    T = np.array([
        [[0.5, 0.0],
         [0.0, 0.5]],
        [[0.0, 0.5],
         [0.5, 0.0]]
    ])
    ghmm = TransitionMatrixGHMM(T)
    
    assert np.isclose(ghmm.word_probability([0, 0]), 0.25)
    assert np.isclose(ghmm.word_probability([0, 1]), 0.25)
    assert np.isclose(ghmm.word_probability([1, 0]), 0.25)
    assert np.isclose(ghmm.word_probability([1, 1]), 0.25)

def test_yield_emissions():
    T = np.array([
        [[0.25, 0.25],
         [0.15, 0.35]],
        [[0.1, 0.4],
         [0.3, 0.2]]
    ])
    ghmm = TransitionMatrixGHMM(T)

    emissions = list(ghmm.yield_emissions(10))
    assert len(emissions) == 10
    assert all(e in [0, 1] for e in emissions)

def test_derive_mixed_state_tree():
    T = np.array([
        [[0.25, 0.25],
         [0.15, 0.35]],
        [[0.1, 0.4],
         [0.3, 0.2]]
    ])
    ghmm = TransitionMatrixGHMM(T)
    tree = ghmm.derive_mixed_state_tree(2)
    assert tree.depth == 2
