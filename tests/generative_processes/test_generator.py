"""Test the generator module."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_factored_process, build_hidden_markov_model
from simplexity.generative_processes.generator import (
    _compute_joint_belief_states,
    generate_data_batch,
    generate_data_batch_with_full_history,
)


def test_generate_data_batch():
    """Test the generate_data_batch function."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    gen_states, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key)
    assert inputs.shape == (batch_size, sequence_len - 1)
    assert labels.shape == (batch_size, sequence_len - 1)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs < hmm.vocab_size)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < hmm.vocab_size)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert isinstance(gen_states, jax.Array)
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_bos_token():
    """Test the generate_data_batch function with a BOS token."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    bos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        bos_token=bos_token,
    )
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs[:, 0] == bos_token)
    assert jnp.all(inputs[:, 1:] < bos_token)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < bos_token)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert isinstance(gen_states, jax.Array)
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_eos_token():
    """Test the generate_data_batch function with an EOS token."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    eos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        eos_token=eos_token,
    )
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs < eos_token)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels[:, :-1] < eos_token)
    assert jnp.all(labels[:, -1] == eos_token)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert isinstance(gen_states, jax.Array)
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_full_history():
    """Ensure belief states and prefix probabilities can be returned."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 4
    sequence_len = 6
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    result = generate_data_batch_with_full_history(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
    )
    # Extract and type-check all fields
    belief_states = result["belief_states"]
    prefix_probs = result["prefix_probabilities"]
    inputs = result["inputs"]
    labels = result["labels"]

    assert isinstance(belief_states, jax.Array)
    assert isinstance(prefix_probs, jax.Array)
    assert isinstance(inputs, jax.Array)
    assert isinstance(labels, jax.Array)

    # Without BOS, belief_states is aligned with inputs (one less than sequence_len)
    assert belief_states.shape == (batch_size, sequence_len - 1, gen_state.shape[0])
    assert prefix_probs.shape == (batch_size, inputs.shape[1])
    assert labels.shape == inputs.shape


def test_generate_data_batch_with_full_history_bos():
    """Ensure belief states align with inputs when BOS token is used."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 4
    sequence_len = 6
    bos_token = 2
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    result = generate_data_batch_with_full_history(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        bos_token=bos_token,
    )
    belief_states = result["belief_states"]
    prefix_probs = result["prefix_probabilities"]
    inputs = result["inputs"]
    labels = result["labels"]

    assert isinstance(belief_states, jax.Array)
    assert isinstance(prefix_probs, jax.Array)
    assert isinstance(inputs, jax.Array)
    assert isinstance(labels, jax.Array)

    # With BOS, inputs has sequence_len positions (BOS + sequence_len-1 tokens)
    # belief_states is aligned with inputs
    assert inputs.shape == (batch_size, sequence_len)
    assert belief_states.shape == (batch_size, sequence_len, gen_state.shape[0])
    assert prefix_probs.shape == (batch_size, inputs.shape[1])
    assert labels.shape == inputs.shape
    # First input should be BOS token
    assert jnp.all(inputs[:, 0] == bos_token)


def _tensor_from_probs(variant_probs):
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


def test_compute_joint_belief_states():
    """Test the outer product computation for joint belief states."""
    factor0 = jnp.array([[[0.6, 0.4], [0.7, 0.3]]], dtype=jnp.float32)
    factor1 = jnp.array([[[0.3, 0.7], [0.5, 0.5]]], dtype=jnp.float32)

    joint = _compute_joint_belief_states((factor0, factor1))

    assert joint.shape == (1, 2, 4)
    expected_t0 = jnp.array([0.6 * 0.3, 0.6 * 0.7, 0.4 * 0.3, 0.4 * 0.7], dtype=jnp.float32)
    expected_t1 = jnp.array([0.7 * 0.5, 0.7 * 0.5, 0.3 * 0.5, 0.3 * 0.5], dtype=jnp.float32)
    chex.assert_trees_all_close(joint[0, 0], expected_t0)
    chex.assert_trees_all_close(joint[0, 1], expected_t1)
    chex.assert_trees_all_close(jnp.sum(joint, axis=-1), jnp.ones((1, 2)))


def test_compute_joint_belief_states_three_factors():
    """Test joint belief computation with three factors."""
    factor0 = jnp.array([[[0.5, 0.5]]], dtype=jnp.float32)
    factor1 = jnp.array([[[0.6, 0.4]]], dtype=jnp.float32)
    factor2 = jnp.array([[[0.3, 0.7]]], dtype=jnp.float32)

    joint = _compute_joint_belief_states((factor0, factor1, factor2))

    assert joint.shape == (1, 1, 8)
    chex.assert_trees_all_close(jnp.sum(joint, axis=-1), jnp.ones((1, 1)))


def test_generate_data_batch_with_full_history_joint_beliefs():
    """Test generating data with joint belief states from factored process."""
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )

    process = build_factored_process(
        structure_type="independent",
        component_types=("hmm", "hmm"),
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
    )

    batch_size = 4
    sequence_len = 6
    gen_state = process.initial_state
    states = tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in gen_state)
    key = jax.random.PRNGKey(42)

    result = generate_data_batch_with_full_history(
        states,
        process,
        batch_size,
        sequence_len,
        key,
        compute_joint_beliefs=True,
    )

    belief_states = result["belief_states"]
    joint_belief_states = result["joint_belief_states"]
    inputs = result["inputs"]

    assert isinstance(belief_states, tuple)
    assert len(belief_states) == 2
    assert isinstance(joint_belief_states, jax.Array)
    assert joint_belief_states.shape == (batch_size, inputs.shape[1], 1 * 1)
    chex.assert_trees_all_close(jnp.sum(joint_belief_states, axis=-1), jnp.ones((batch_size, inputs.shape[1])))


def test_generate_data_batch_with_full_history_no_joint_beliefs():
    """Test that joint_belief_states is not included when compute_joint_beliefs=False."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 4
    sequence_len = 6
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)

    result = generate_data_batch_with_full_history(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        compute_joint_beliefs=False,
    )

    assert "joint_belief_states" not in result


def test_generate_data_batch_with_full_history_joint_beliefs_non_factored():
    """Test that joint_belief_states is not included for non-factored processes."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 4
    sequence_len = 6
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)

    result = generate_data_batch_with_full_history(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        compute_joint_beliefs=True,
    )

    assert "joint_belief_states" not in result


def test_generate_data_batch_with_full_history_joint_beliefs_values():
    """Verify joint belief states match manual outer product of factored states."""
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.3, 0.7]]),
        _tensor_from_probs([[0.8, 0.2], [0.4, 0.6]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((1, 2), dtype=jnp.float32),
        jnp.ones((1, 2), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([0.5, 0.5], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    )

    process = build_factored_process(
        structure_type="independent",
        component_types=("hmm", "hmm"),
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
    )

    batch_size = 2
    sequence_len = 4
    gen_state = process.initial_state
    states = tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in gen_state)
    key = jax.random.PRNGKey(123)

    result = generate_data_batch_with_full_history(
        states,
        process,
        batch_size,
        sequence_len,
        key,
        compute_joint_beliefs=True,
    )

    belief_states = result["belief_states"]
    joint_belief_states = result["joint_belief_states"]

    assert isinstance(belief_states, tuple)
    assert joint_belief_states.shape == (batch_size, result["inputs"].shape[1], 4)

    expected_joint = _compute_joint_belief_states(belief_states)
    chex.assert_trees_all_close(joint_belief_states, expected_joint)
