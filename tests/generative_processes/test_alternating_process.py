"""Tests for simple (non-factored) alternating process."""

import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.alternating_process import AlternatingProcess
from simplexity.generative_processes.builder import (
    build_simple_alternating_process,
    build_simple_alternating_process_from_spec,
)


def test_alternating_process_init():
    """Test AlternatingProcess initialization."""
    n, V, S = 2, 3, 4
    T = jnp.ones((n, V, S, S)) * 0.5
    norm = jnp.ones((n, S))
    initial_state = jnp.ones(S) / S

    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=norm,
        initial_state=initial_state,
        n_repetitions=1,
    )

    assert process.vocab_size == V
    assert process.num_states == S
    assert process.n == n
    assert process.n_repetitions == 1


def test_alternating_process_invalid_shape():
    """Test that invalid transition matrix shapes are rejected."""
    # Wrong number of dimensions
    with pytest.raises(ValueError, match="must have shape"):
        AlternatingProcess(
            component_type="hmm",
            transition_matrices=jnp.ones((2, 3, 4)),  # Missing a dimension
            normalizing_eigenvectors=jnp.ones((2, 4)),
            initial_state=jnp.ones(4) / 4,
        )


def test_alternating_process_invalid_state_dim():
    """Test mismatched state dimensions."""
    with pytest.raises(ValueError, match="State dimensions must match"):
        AlternatingProcess(
            component_type="hmm",
            transition_matrices=jnp.ones((2, 3, 4, 5)),  # S1=4, S2=5 mismatch
            normalizing_eigenvectors=jnp.ones((2, 4)),
            initial_state=jnp.ones(4) / 4,
        )


def test_alternating_process_invalid_initial_state():
    """Test mismatched initial state shape."""
    with pytest.raises(ValueError, match="initial_state shape"):
        AlternatingProcess(
            component_type="hmm",
            transition_matrices=jnp.ones((2, 3, 4, 4)),
            normalizing_eigenvectors=jnp.ones((2, 4)),
            initial_state=jnp.ones(5) / 5,  # Wrong size
        )


def test_alternating_process_invalid_repetitions():
    """Test invalid n_repetitions."""
    with pytest.raises(ValueError, match="n_repetitions must be positive"):
        AlternatingProcess(
            component_type="hmm",
            transition_matrices=jnp.ones((2, 3, 4, 4)),
            normalizing_eigenvectors=jnp.ones((2, 4)),
            initial_state=jnp.ones(4) / 4,
            n_repetitions=0,
        )


def test_alternating_process_initial_state():
    """Test initial state structure."""
    n, V, S = 2, 3, 4
    T = jnp.ones((n, V, S, S)) * 0.5
    initial_dist = jnp.ones(S) / S

    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=initial_dist,
    )

    position, belief_state = process.initial_state
    assert position == 0
    assert belief_state.shape == (S,)
    assert jnp.allclose(belief_state, initial_dist)


def test_alternating_process_variant_selection():
    """Test variant selection logic."""
    n, V, S = 3, 2, 2
    n_repetitions = 2

    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
        n_repetitions=n_repetitions,
    )

    # Positions 0,1 -> variant 0
    assert int(process._select_variant(jnp.array(0))) == 0
    assert int(process._select_variant(jnp.array(1))) == 0

    # Positions 2,3 -> variant 1
    assert int(process._select_variant(jnp.array(2))) == 1
    assert int(process._select_variant(jnp.array(3))) == 1

    # Positions 4,5 -> variant 2
    assert int(process._select_variant(jnp.array(4))) == 2
    assert int(process._select_variant(jnp.array(5))) == 2

    # Position 6 -> wraps to variant 0
    assert int(process._select_variant(jnp.array(6))) == 0


def test_alternating_process_observation_distribution():
    """Test observation probability distribution."""
    n, V, S = 2, 3, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    state = process.initial_state
    dist = process.observation_probability_distribution(state)

    assert dist.shape == (V,)
    assert jnp.all(dist >= 0)


def test_alternating_process_emission():
    """Test observation emission."""
    n, V, S = 2, 3, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    key = jax.random.PRNGKey(42)
    state = process.initial_state
    obs = process.emit_observation(state, key)

    assert obs.shape == ()
    assert 0 <= obs < V


def test_alternating_process_transition():
    """Test state transition."""
    n, V, S = 2, 3, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    state = process.initial_state
    position, belief_state = state

    assert position == 0

    # Transition
    new_state = process.transition_states(state, jnp.array(0))
    new_position, new_belief_state = new_state

    assert new_position == 1
    assert new_belief_state.shape == (S,)


def test_alternating_process_position_increment():
    """Test position increments through sequence."""
    n, V, S = 2, 2, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    state = process.initial_state

    for expected_pos in range(5):
        position, _ = state
        assert position == expected_pos
        state = process.transition_states(state, jnp.array(0))


def test_alternating_process_probability():
    """Test probability computation."""
    n, V, S = 2, 2, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    observations = jnp.array([0, 1, 0, 1])
    prob = process.probability(observations)

    assert prob.shape == ()
    assert 0 <= prob <= 1


def test_alternating_process_log_probability():
    """Test log probability computation."""
    n, V, S = 2, 2, 2
    T = jnp.ones((n, V, S, S)) * 0.5
    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
    )

    observations = jnp.array([0, 1, 0])
    log_prob = process.log_probability(observations)

    assert log_prob.shape == ()
    assert log_prob <= 0


def test_build_simple_alternating_process():
    """Test builder function."""
    n, V, S = 2, 3, 4
    T = jnp.ones((n, V, S, S)) * 0.5

    process = build_simple_alternating_process(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.ones(S) / S,
        n_repetitions=2,
    )

    assert isinstance(process, AlternatingProcess)
    assert process.vocab_size == V
    assert process.num_states == S
    assert process.n == n
    assert process.n_repetitions == 2


def test_build_simple_alternating_process_from_spec():
    """Test spec-based builder."""
    variants = [
        {"process_name": "coin", "p": 0.7},
        {"process_name": "coin", "p": 0.3},
    ]

    process = build_simple_alternating_process_from_spec(
        variants=variants,
        component_type="hmm",
        n_repetitions=3,
    )

    assert isinstance(process, AlternatingProcess)
    assert process.n == 2
    assert process.n_repetitions == 3
    assert process.vocab_size == 2  # coin has 2 tokens


def test_build_simple_alternating_process_from_spec_empty():
    """Test that empty variants list is rejected."""
    with pytest.raises(ValueError, match="Must provide at least one variant"):
        build_simple_alternating_process_from_spec(
            variants=[],
            component_type="hmm",
        )


def test_build_simple_alternating_process_from_spec_mismatched_shapes():
    """Test that mismatched variant shapes are rejected."""
    variants = [
        {"process_name": "coin", "p": 0.7},  # 2 tokens
        {"process_name": "mess3", "x": 0.5, "a": 0.6},  # 3 tokens
    ]

    with pytest.raises(ValueError, match="All variants must have same shape"):
        build_simple_alternating_process_from_spec(
            variants=variants,
            component_type="hmm",
        )


def test_alternating_process_with_repetitions_full_cycle():
    """Test full alternating cycle with repetitions."""
    n = 3
    n_repetitions = 2
    S = 2

    # Create distinct matrices for each variant (deterministic for testing)
    T0 = jnp.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    T1 = jnp.array([[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]])
    T2 = jnp.array([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    T = jnp.stack([T0, T1, T2])

    process = AlternatingProcess(
        component_type="hmm",
        transition_matrices=T,
        normalizing_eigenvectors=jnp.ones((n, S)),
        initial_state=jnp.array([1.0, 0.0]),
        n_repetitions=n_repetitions,
    )

    state = process.initial_state

    # Verify cycling pattern: 0,0,1,1,2,2,0,0,...
    expected_variants = [0, 0, 1, 1, 2, 2, 0, 0]

    for i, expected_variant in enumerate(expected_variants):
        position, _ = state
        assert position == i
        actual_variant = int(process._select_variant(position))
        assert actual_variant == expected_variant, f"Position {i} should use variant {expected_variant}"

        # Advance state
        if i < len(expected_variants) - 1:
            state = process.transition_states(state, jnp.array(0))
