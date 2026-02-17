"""Tests for InflatedVocabularyProcess."""

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

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
    build_inflated_process,
    build_inflated_process_from_spec,
)
from simplexity.generative_processes.inflated_vocabulary_process import InflatedVocabularyProcess
from simplexity.generative_processes.nonergodic_generative_process import (
    NonErgodicGenerativeProcess,
    NonErgodicState,
)


class TestBasicProperties:
    """Tests for basic properties of InflatedVocabularyProcess."""

    @pytest.fixture
    def coin_k3(self):
        """Coin process with K=3 inflation."""
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        return InflatedVocabularyProcess(coin, inflation_factor=3)

    @pytest.fixture
    def mess3_k3(self):
        """Mess3 process with K=3 inflation."""
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        return InflatedVocabularyProcess(mess3, inflation_factor=3)

    def test_vocab_size_coin(self, coin_k3: InflatedVocabularyProcess):
        assert coin_k3.vocab_size == 6

    def test_vocab_size_mess3(self, mess3_k3: InflatedVocabularyProcess):
        assert mess3_k3.vocab_size == 9

    def test_initial_state_matches_base(self, coin_k3: InflatedVocabularyProcess):
        chex.assert_trees_all_close(coin_k3.initial_state, coin_k3.base_process.initial_state)

    def test_inflation_factor_stored(self, coin_k3: InflatedVocabularyProcess):
        assert coin_k3.inflation_factor == 3

    def test_invalid_inflation_factor_raises(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        with pytest.raises(ValueError, match="inflation_factor must be >= 2"):
            InflatedVocabularyProcess(coin, inflation_factor=1)

    def test_invalid_inflation_factor_zero_raises(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        with pytest.raises(ValueError, match="inflation_factor must be >= 2"):
            InflatedVocabularyProcess(coin, inflation_factor=0)


class TestObservationDistribution:
    """Tests for observation probability distribution."""

    @pytest.fixture
    def coin_k3(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        return InflatedVocabularyProcess(coin, inflation_factor=3)

    def test_distribution_sums_to_one(self, coin_k3: InflatedVocabularyProcess):
        state = coin_k3.initial_state
        dist = coin_k3.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_distribution_has_correct_size(self, coin_k3: InflatedVocabularyProcess):
        state = coin_k3.initial_state
        dist = coin_k3.observation_probability_distribution(state)
        assert dist.shape == (6,)

    def test_distribution_spreads_uniformly(self, coin_k3: InflatedVocabularyProcess):
        """Each base token's prob is split equally among K noise variants."""
        state = coin_k3.initial_state
        dist = coin_k3.observation_probability_distribution(state)
        expected = jnp.array([0.7 / 3, 0.3 / 3, 0.7 / 3, 0.3 / 3, 0.7 / 3, 0.3 / 3])
        chex.assert_trees_all_close(dist, expected, atol=1e-6)

    def test_noise_variants_have_equal_probability(self):
        """All K noise variants of the same base token should have identical probability."""
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        inflated = InflatedVocabularyProcess(mess3, inflation_factor=4)
        state = inflated.initial_state
        dist = inflated.observation_probability_distribution(state)
        v_base = mess3.vocab_size
        for base_tok in range(v_base):
            probs = [float(dist[n * v_base + base_tok]) for n in range(4)]
            for p in probs[1:]:
                chex.assert_trees_all_close(p, probs[0], atol=1e-6)

    def test_log_distribution_consistent(self, coin_k3: InflatedVocabularyProcess):
        state = coin_k3.initial_state
        log_state = jnp.log(state)
        dist = coin_k3.observation_probability_distribution(state)
        log_dist = coin_k3.log_observation_probability_distribution(log_state)
        chex.assert_trees_all_close(log_dist, jnp.log(dist), atol=1e-5)


class TestTransitionStates:
    """Tests for state transitions."""

    def test_noise_prefix_does_not_affect_state(self):
        """All K noise variants of the same base token should produce identical states."""
        even_ones = build_hidden_markov_model("even_ones", {"p": 0.5})
        inflated = InflatedVocabularyProcess(even_ones, inflation_factor=3)
        state = inflated.initial_state
        v_base = even_ones.vocab_size

        for base_tok in range(v_base):
            states = [inflated.transition_states(state, jnp.array(n * v_base + base_tok)) for n in range(3)]
            for s in states[1:]:
                chex.assert_trees_all_close(s, states[0], atol=1e-6)

    def test_transition_matches_base_process(self):
        """Transitioning with an inflated token should match transitioning with the base token."""
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        inflated = InflatedVocabularyProcess(mess3, inflation_factor=3)
        state = mess3.initial_state

        for base_tok in range(mess3.vocab_size):
            base_state = mess3.transition_states(state, jnp.array(base_tok))
            inflated_state = inflated.transition_states(state, jnp.array(base_tok))
            chex.assert_trees_all_close(inflated_state, base_state, atol=1e-6)

            inflated_state_noisy = inflated.transition_states(state, jnp.array(2 * mess3.vocab_size + base_tok))
            chex.assert_trees_all_close(inflated_state_noisy, base_state, atol=1e-6)


class TestProbability:
    """Tests for sequence probability computation."""

    def test_probability_scales_by_inflation_penalty(self):
        """P(inflated_seq) = P(base_seq) / K^T."""
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        k = 3
        inflated = InflatedVocabularyProcess(coin, inflation_factor=k)

        base_seq = jnp.array([0, 1, 0])
        base_prob = coin.probability(base_seq)
        inflated_prob = inflated.probability(base_seq)

        expected = base_prob / (k**3)
        chex.assert_trees_all_close(inflated_prob, expected, atol=1e-6)

    def test_probability_same_base_different_noise(self):
        """Different noise prefixes with same base sequence should have same probability."""
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        inflated = InflatedVocabularyProcess(coin, inflation_factor=3)

        seq_noise0 = jnp.array([0, 1, 0])
        seq_noise1 = jnp.array([2, 3, 2])
        seq_noise2 = jnp.array([4, 5, 4])

        p0 = inflated.probability(seq_noise0)
        p1 = inflated.probability(seq_noise1)
        p2 = inflated.probability(seq_noise2)

        chex.assert_trees_all_close(p0, p1, atol=1e-6)
        chex.assert_trees_all_close(p1, p2, atol=1e-6)

    def test_log_probability_consistent(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        inflated = InflatedVocabularyProcess(coin, inflation_factor=3)

        seq = jnp.array([0, 3, 1, 4])
        prob = inflated.probability(seq)
        log_prob = inflated.log_probability(seq)
        chex.assert_trees_all_close(log_prob, jnp.log(prob), atol=1e-5)

    def test_optimal_loss_increases_by_log_k(self):
        """Average per-token loss should increase by exactly log(K)."""
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        k = 4
        inflated = InflatedVocabularyProcess(mess3, inflation_factor=k)

        key = jax.random.PRNGKey(42)
        state = mess3.initial_state
        batch_state = jnp.broadcast_to(state, (100,) + state.shape)
        keys = jax.random.split(key, 100)

        _, base_seqs = mess3.generate(batch_state, keys, 200, False)
        base_log_probs = jax.vmap(mess3.log_probability)(base_seqs)
        base_avg_loss = -jnp.mean(base_log_probs) / 200

        inflated_log_probs = jax.vmap(inflated.log_probability)(base_seqs)
        inflated_avg_loss = -jnp.mean(inflated_log_probs) / 200

        expected_increase = jnp.log(jnp.array(k, dtype=jnp.float32))
        chex.assert_trees_all_close(inflated_avg_loss - base_avg_loss, expected_increase, atol=0.01)


class TestGeneration:
    """Tests for sequence generation."""

    def test_generate_valid_tokens(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        inflated = InflatedVocabularyProcess(coin, inflation_factor=3)

        state = inflated.initial_state
        batch_state = jnp.broadcast_to(state, (4,) + state.shape)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        _, observations = inflated.generate(batch_state, keys, 20, False)

        assert observations.shape == (4, 20)
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < inflated.vocab_size)

    def test_generate_covers_noise_variants(self):
        """Generated tokens should use all noise variants over many samples."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})
        inflated = InflatedVocabularyProcess(coin, inflation_factor=3)

        state = inflated.initial_state
        batch_state = jnp.broadcast_to(state, (50,) + state.shape)
        keys = jax.random.split(jax.random.PRNGKey(123), 50)
        _, observations = inflated.generate(batch_state, keys, 100, False)

        unique_tokens = jnp.unique(observations.ravel())
        assert unique_tokens.shape[0] == 6

    def test_generate_with_return_all_states(self):
        """Generation with return_all_states=True should return state trajectory."""
        mess3 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        inflated = InflatedVocabularyProcess(mess3, inflation_factor=2)

        state = inflated.initial_state
        batch_state = jnp.broadcast_to(state, (4,) + state.shape)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states, observations = inflated.generate(batch_state, keys, 10, True)

        assert observations.shape == (4, 10)
        assert states.shape == (4, 10) + state.shape

    def test_base_token_distribution_matches(self):
        """Extracting base tokens from inflated generation should match base distribution."""
        coin = build_hidden_markov_model("coin", {"p": 0.8})
        inflated = InflatedVocabularyProcess(coin, inflation_factor=5)

        state = inflated.initial_state
        batch_state = jnp.broadcast_to(state, (200,) + state.shape)
        keys = jax.random.split(jax.random.PRNGKey(42), 200)
        _, observations = inflated.generate(batch_state, keys, 500, False)

        base_tokens = observations % coin.vocab_size
        base_freq = jnp.mean(base_tokens == 0)
        chex.assert_trees_all_close(base_freq, 0.8, atol=0.03)


class TestWithDifferentBaseProcesses:
    """Tests for wrapping different process types."""

    def test_wrap_ghmm(self):
        ghmm = build_generalized_hidden_markov_model("tom_quantum", {"alpha": 1.0, "beta": 1.0})
        inflated = InflatedVocabularyProcess(ghmm, inflation_factor=2)
        assert inflated.vocab_size == 2 * ghmm.vocab_size

        state = inflated.initial_state
        dist = inflated.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

        batch_state = jnp.broadcast_to(state, (4,) + state.shape)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        _, observations = inflated.generate(batch_state, keys, 10, False)
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < inflated.vocab_size)

    def test_wrap_nonergodic(self):
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})
        nonergodic = NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[0.5, 0.5],
        )
        inflated = InflatedVocabularyProcess(nonergodic, inflation_factor=3)
        assert inflated.vocab_size == 6

        state = inflated.initial_state
        assert isinstance(state, NonErgodicState)
        dist = inflated.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_double_inflation(self):
        """Stacking inflation: K1 * K2 total inflation."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})
        inflated1 = InflatedVocabularyProcess(coin, inflation_factor=2)
        inflated2 = InflatedVocabularyProcess(inflated1, inflation_factor=3)
        assert inflated2.vocab_size == 12

        state = inflated2.initial_state
        dist = inflated2.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)
        chex.assert_trees_all_close(dist, jnp.ones(12) / 12, atol=1e-6)


class TestBuilder:
    """Tests for builder functions."""

    def test_build_inflated_process(self):
        coin = build_hidden_markov_model("coin", {"p": 0.7})
        inflated = build_inflated_process(coin, inflation_factor=3)
        assert isinstance(inflated, InflatedVocabularyProcess)
        assert inflated.vocab_size == 6

    def test_build_inflated_process_from_spec_hmm(self):
        inflated = build_inflated_process_from_spec(
            base_spec={
                "component_type": "hmm",
                "process_name": "mess3",
                "process_params": {"x": 0.15, "a": 0.6},
            },
            inflation_factor=3,
        )
        assert isinstance(inflated, InflatedVocabularyProcess)
        assert inflated.vocab_size == 9

    def test_build_inflated_process_from_spec_ghmm(self):
        inflated = build_inflated_process_from_spec(
            base_spec={
                "component_type": "ghmm",
                "process_name": "tom_quantum",
                "process_params": {"alpha": 1.0, "beta": 1.0},
            },
            inflation_factor=2,
        )
        assert isinstance(inflated, InflatedVocabularyProcess)
        state = inflated.initial_state
        dist = inflated.observation_probability_distribution(state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_build_inflated_process_from_spec_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown base component_type"):
            build_inflated_process_from_spec(
                base_spec={"component_type": "unknown", "process_name": "coin"},
                inflation_factor=2,
            )
