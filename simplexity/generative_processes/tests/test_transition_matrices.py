from simplexity.generative_processes.transition_matrices import (
    days_of_week,
    fanizza,
    mess3,
    post_quantum,
    rrxor,
    tom_quantum,
)


def test_post_quantum():
    transition_matrices = post_quantum()
    assert transition_matrices.shape == (3, 3, 3)


def test_days_of_week():
    transition_matrices = days_of_week()
    assert transition_matrices.shape == (11, 7, 7)


def test_tom_quantum():
    transition_matrices = tom_quantum(alpha=1.0, beta=1.0)
    assert transition_matrices.shape == (4, 3, 3)


def test_fanizza():
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    assert transition_matrices.shape == (2, 4, 4)


def test_rrxor():
    transition_matrices = rrxor()
    assert transition_matrices.shape == (2, 5, 5)


def test_mess3():
    transition_matrices = mess3()
    assert transition_matrices.shape == (3, 3, 3)