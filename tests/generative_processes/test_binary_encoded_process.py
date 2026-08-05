"""Tests for BinaryEncodedProcess."""

# pylint: disable-all
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
import pytest

from simplexity.generative_processes.binary_encoded_process import BinaryEncodedProcess, BinaryEncodedState
from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
)


class TestBasicProperties:
    """Tests for basic properties of BinaryEncodedProcess."""

    @pytest.fixture
    def mess3_binary(self) -> BinaryEncodedProcess:
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        return BinaryEncodedProcess(mess3)

    @pytest.fixture
    def coin_binary(self) -> BinaryEncodedProcess:
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        return BinaryEncodedProcess(coin)

    def test_vocab_size_is_two(self, mess3_binary: BinaryEncodedProcess):
        assert mess3_binary.vocab_size == 2

    def test_num_bits_mess3(self, mess3_binary: BinaryEncodedProcess):
        assert mess3_binary.num_bits == 2

    def test_num_bits_coin(self, coin_binary: BinaryEncodedProcess):
        assert coin_binary.num_bits == 1

    def test_initial_state_base_matches(self, mess3_binary: BinaryEncodedProcess):
        chex.assert_trees_all_close(
            mess3_binary.initial_state.base_state,
            mess3_binary.base_process.initial_state,
        )

    def test_initial_state_bit_position_is_zero(self, mess3_binary: BinaryEncodedProcess):
        assert mess3_binary.initial_state.bit_position == 0

    def test_initial_state_accumulated_prefix_is_zero(self, mess3_binary: BinaryEncodedProcess):
        assert mess3_binary.initial_state.accumulated_prefix == 0


class TestObservationDistribution:
    """Tests for observation probability distribution."""

    @pytest.fixture
    def mess3_binary(self) -> BinaryEncodedProcess:
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        return BinaryEncodedProcess(mess3)

    def test_distribution_sums_to_one(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        dist = mess3_binary.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_distribution_has_correct_size(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        dist = mess3_binary.observation_probability_distribution(state)
        assert dist.shape == (2,)

    def test_first_bit_marginalizes_correctly(self, mess3_binary: BinaryEncodedProcess):
        """First bit: P(0) = P(token 0) + P(token 1), P(1) = P(token 2)."""
        state = mess3_binary.initial_state
        base_dist = mess3_binary.base_process.observation_probability_distribution(state.base_state)
        binary_dist = mess3_binary.observation_probability_distribution(state)

        expected_p0 = base_dist[0] + base_dist[1]
        expected_p1 = base_dist[2]
        chex.assert_trees_all_close(binary_dist[0], expected_p0, atol=1e-6)
        chex.assert_trees_all_close(binary_dist[1], expected_p1, atol=1e-6)

    def test_second_bit_given_first_zero(self, mess3_binary: BinaryEncodedProcess):
        """P(bit=0 | first=0) = P(token 0) / (P(token 0) + P(token 1))."""
        state = mess3_binary.initial_state
        base_dist = mess3_binary.base_process.observation_probability_distribution(state.base_state)

        state_after_0 = BinaryEncodedState(
            base_state=state.base_state,
            bit_position=jnp.array(1, dtype=jnp.int32),
            accumulated_prefix=jnp.array(0, dtype=jnp.int32),
        )
        dist = mess3_binary.observation_probability_distribution(state_after_0)

        denom = base_dist[0] + base_dist[1]
        chex.assert_trees_all_close(dist[0], base_dist[0] / denom, atol=1e-6)
        chex.assert_trees_all_close(dist[1], base_dist[1] / denom, atol=1e-6)

    def test_second_bit_given_first_one_is_deterministic(self, mess3_binary: BinaryEncodedProcess):
        """P(bit=0 | first=1) = 1.0 since code 11 is unused for mess3."""
        state = mess3_binary.initial_state

        state_after_1 = BinaryEncodedState(
            base_state=state.base_state,
            bit_position=jnp.array(1, dtype=jnp.int32),
            accumulated_prefix=jnp.array(1, dtype=jnp.int32),
        )
        dist = mess3_binary.observation_probability_distribution(state_after_1)

        chex.assert_trees_all_close(dist[0], 1.0, atol=1e-6)
        chex.assert_trees_all_close(dist[1], 0.0, atol=1e-6)

    def test_distribution_sums_to_one_at_intermediate_position(self, mess3_binary: BinaryEncodedProcess):
        state_after_0 = BinaryEncodedState(
            base_state=mess3_binary.initial_state.base_state,
            bit_position=jnp.array(1, dtype=jnp.int32),
            accumulated_prefix=jnp.array(0, dtype=jnp.int32),
        )
        dist = mess3_binary.observation_probability_distribution(state_after_0)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_log_distribution_consistent(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        log_state = BinaryEncodedState(
            base_state=jnp.log(state.base_state),
            bit_position=state.bit_position,
            accumulated_prefix=state.accumulated_prefix,
        )
        dist = mess3_binary.observation_probability_distribution(state)
        log_dist = mess3_binary.log_observation_probability_distribution(log_state)
        chex.assert_trees_all_close(log_dist, jnp.log(dist), atol=1e-5)


class TestTransitionStates:
    """Tests for state transitions."""

    @pytest.fixture
    def mess3_binary(self) -> BinaryEncodedProcess:
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        return BinaryEncodedProcess(mess3)

    def test_base_state_unchanged_during_partial_token(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        new_state = mess3_binary.transition_states(state, jnp.array(0))
        chex.assert_trees_all_close(new_state.base_state, state.base_state, atol=1e-6)

    def test_bit_position_increments(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        new_state = mess3_binary.transition_states(state, jnp.array(0))
        assert new_state.bit_position == 1

    def test_accumulated_prefix_updates(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        new_state = mess3_binary.transition_states(state, jnp.array(1))
        assert new_state.accumulated_prefix == 1

    def test_base_state_transitions_after_complete_token(self, mess3_binary: BinaryEncodedProcess):
        """After emitting bits 01 (token 1), base state should match base transition with token 1."""
        state = mess3_binary.initial_state
        state_after_0 = mess3_binary.transition_states(state, jnp.array(0))
        state_after_01 = mess3_binary.transition_states(state_after_0, jnp.array(1))

        expected_base = mess3_binary.base_process.transition_states(state.base_state, jnp.array(1))
        chex.assert_trees_all_close(state_after_01.base_state, expected_base, atol=1e-6)

    def test_position_resets_after_complete_token(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        state_after_0 = mess3_binary.transition_states(state, jnp.array(0))
        state_after_01 = mess3_binary.transition_states(state_after_0, jnp.array(1))
        assert state_after_01.bit_position == 0

    def test_prefix_resets_after_complete_token(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        state_after_0 = mess3_binary.transition_states(state, jnp.array(0))
        state_after_01 = mess3_binary.transition_states(state_after_0, jnp.array(1))
        assert state_after_01.accumulated_prefix == 0

    def test_token_0_transition(self, mess3_binary: BinaryEncodedProcess):
        """Bits 00 should decode to token 0."""
        state = mess3_binary.initial_state
        state_after_0 = mess3_binary.transition_states(state, jnp.array(0))
        state_after_00 = mess3_binary.transition_states(state_after_0, jnp.array(0))

        expected = mess3_binary.base_process.transition_states(state.base_state, jnp.array(0))
        chex.assert_trees_all_close(state_after_00.base_state, expected, atol=1e-6)

    def test_token_2_transition(self, mess3_binary: BinaryEncodedProcess):
        """Bits 10 should decode to token 2."""
        state = mess3_binary.initial_state
        state_after_1 = mess3_binary.transition_states(state, jnp.array(1))
        state_after_10 = mess3_binary.transition_states(state_after_1, jnp.array(0))

        expected = mess3_binary.base_process.transition_states(state.base_state, jnp.array(2))
        chex.assert_trees_all_close(state_after_10.base_state, expected, atol=1e-6)


class TestProbability:
    """Tests for sequence probability computation."""

    @pytest.fixture
    def mess3(self):
        return build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})

    @pytest.fixture
    def mess3_binary(self, mess3) -> BinaryEncodedProcess:
        return BinaryEncodedProcess(mess3)

    def test_complete_sequence_matches_base(self, mess3, mess3_binary: BinaryEncodedProcess):
        """Binary sequence probability equals base sequence probability for complete tokens."""
        base_seq = jnp.array([0, 1, 2, 0])
        binary_seq = jnp.array([0, 0, 0, 1, 1, 0, 0, 0])

        base_prob = mess3.probability(base_seq)
        binary_prob = mess3_binary.probability(binary_seq)
        chex.assert_trees_all_close(binary_prob, base_prob, atol=1e-6)

    def test_incomplete_sequence_returns_valid_probability(self, mess3_binary: BinaryEncodedProcess):
        binary_seq = jnp.array([0, 0, 0])
        prob = mess3_binary.probability(binary_seq)
        assert prob > 0
        assert prob <= 1

    def test_incomplete_sequence_extends_complete(self, mess3_binary: BinaryEncodedProcess):
        """P(b0, b1, b2) = P(b0, b1) * P(b2 | b0, b1) and both should be valid."""
        complete = jnp.array([0, 0])
        incomplete = jnp.array([0, 0, 0])
        p_complete = mess3_binary.probability(complete)
        p_incomplete = mess3_binary.probability(incomplete)
        assert p_incomplete <= p_complete

    def test_log_probability_consistent(self, mess3_binary: BinaryEncodedProcess):
        binary_seq = jnp.array([0, 1, 1, 0, 0, 0])
        prob = mess3_binary.probability(binary_seq)
        log_prob = mess3_binary.log_probability(binary_seq)
        chex.assert_trees_all_close(log_prob, jnp.log(prob), atol=1e-5)

    def test_multiple_complete_sequences(self, mess3, mess3_binary: BinaryEncodedProcess):
        """Verify several different token sequences."""
        for base_tokens in [[0], [1], [2], [2, 1, 0], [1, 1, 1]]:
            base_seq = jnp.array(base_tokens)
            bits = []
            for t in base_tokens:
                bits.extend([(t >> 1) & 1, t & 1])
            binary_seq = jnp.array(bits)

            base_prob = mess3.probability(base_seq)
            binary_prob = mess3_binary.probability(binary_seq)
            chex.assert_trees_all_close(binary_prob, base_prob, atol=1e-6)


class TestGeneration:
    """Tests for sequence generation."""

    @pytest.fixture
    def mess3_binary(self) -> BinaryEncodedProcess:
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        return BinaryEncodedProcess(mess3)

    def test_generate_valid_tokens(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        batch_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape), state)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        _, observations = mess3_binary.generate(batch_state, keys, 20, False)

        assert observations.shape == (4, 20)
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < 2)

    def test_generate_with_return_all_states(self, mess3_binary: BinaryEncodedProcess):
        state = mess3_binary.initial_state
        batch_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape), state)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states, observations = mess3_binary.generate(batch_state, keys, 10, True)

        assert observations.shape == (4, 10)
        assert states.base_state.shape == (4, 10) + mess3_binary.base_process.initial_state.shape

    def test_decoded_tokens_are_valid(self, mess3_binary: BinaryEncodedProcess):
        """All decoded binary pairs should map to valid tokens (0, 1, or 2)."""
        state = mess3_binary.initial_state
        batch_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (50,) + x.shape), state)
        keys = jax.random.split(jax.random.PRNGKey(42), 50)
        _, observations = mess3_binary.generate(batch_state, keys, 200, False)

        pairs = observations.reshape(50, 100, 2)
        decoded = pairs[:, :, 0] * 2 + pairs[:, :, 1]
        assert jnp.all(decoded < 3)

    def test_decoded_generation_matches_base_distribution(self, mess3_binary: BinaryEncodedProcess):
        """Decoded token frequencies should approximate the stationary distribution."""
        state = mess3_binary.initial_state
        batch_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (200,) + x.shape), state)
        keys = jax.random.split(jax.random.PRNGKey(42), 200)
        _, observations = mess3_binary.generate(batch_state, keys, 200, False)

        pairs = observations.reshape(200, 100, 2)
        decoded = pairs[:, :, 0] * 2 + pairs[:, :, 1]

        for token in range(3):
            freq = jnp.mean(decoded == token)
            chex.assert_trees_all_close(freq, 1.0 / 3.0, atol=0.05)


class TestWithDifferentBaseProcesses:
    """Tests for wrapping different process types."""

    def test_coin_has_one_bit(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        binary = BinaryEncodedProcess(coin)
        assert binary.num_bits == 1

    def test_coin_binary_encoding_is_identity(self):
        """With vocab_size=2, binary encoding should reproduce the base distribution exactly."""
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        binary = BinaryEncodedProcess(coin)

        state = binary.initial_state
        dist = binary.observation_probability_distribution(state)
        base_dist = coin.observation_probability_distribution(coin.initial_state)
        chex.assert_trees_all_close(dist, base_dist, atol=1e-6)

    def test_coin_probability_matches_base(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        binary = BinaryEncodedProcess(coin)

        seq = jnp.array([0, 1, 0, 0, 1])
        chex.assert_trees_all_close(binary.probability(seq), coin.probability(seq), atol=1e-6)

    def test_wrap_ghmm(self):
        ghmm = build_generalized_hidden_markov_model("tom_quantum", {"alpha": 1.0, "beta": 1.0})
        binary = BinaryEncodedProcess(ghmm)
        assert binary.vocab_size == 2

        state = binary.initial_state
        dist = binary.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_ghmm_generation(self):
        ghmm = build_generalized_hidden_markov_model("tom_quantum", {"alpha": 1.0, "beta": 1.0})
        binary = BinaryEncodedProcess(ghmm)

        state = binary.initial_state
        batch_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape), state)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        _, observations = binary.generate(batch_state, keys, 10, False)

        assert jnp.all(observations >= 0)
        assert jnp.all(observations < 2)
