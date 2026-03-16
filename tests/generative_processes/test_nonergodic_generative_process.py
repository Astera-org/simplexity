"""Tests for NonErgodicGenerativeProcess."""

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
    build_factored_process_from_spec,
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
    build_nonergodic_process_from_spec,
)
from simplexity.generative_processes.generator import generate_data_batch_with_full_history
from simplexity.generative_processes.nonergodic_generative_process import (
    ComponentState,
    NonErgodicGenerativeProcess,
    NonErgodicState,
)


def _expand_component_state(
    state: ComponentState,
    batch_size: int,
) -> ComponentState:
    """Expand a single component state to a batch of identical states."""
    if isinstance(state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in state)
    return jnp.repeat(state[None, :], batch_size, axis=0)


def _expand_state(state: NonErgodicState, batch_size: int) -> NonErgodicState:
    """Expand a single NonErgodicState to a batch of identical states."""
    return NonErgodicState(
        component_beliefs=jnp.repeat(state.component_beliefs[None, :], batch_size, axis=0),
        component_states=tuple(_expand_component_state(cs, batch_size) for cs in state.component_states),
    )


class TestNonErgodicState:
    """Tests for NonErgodicState structure."""

    def test_state_is_named_tuple(self):
        """NonErgodicState should be a NamedTuple with named fields."""
        state = NonErgodicState(
            component_beliefs=jnp.array([0.5, 0.5]),
            component_states=(jnp.array([1.0, 0.0]), jnp.array([0.5, 0.5])),
        )
        assert hasattr(state, "component_beliefs")
        assert hasattr(state, "component_states")
        assert isinstance(state, tuple)

    def test_state_is_pytree_compatible(self):
        """NonErgodicState should be compatible with JAX pytree operations."""
        state = NonErgodicState(
            component_beliefs=jnp.array([0.5, 0.5]),
            component_states=(jnp.array([1.0, 0.0]), jnp.array([0.5, 0.5])),
        )
        # Should work with tree_map
        doubled = jax.tree_util.tree_map(lambda x: x * 2, state)
        chex.assert_trees_all_close(doubled.component_beliefs, jnp.array([1.0, 1.0]))


class TestNonErgodicGenerativeProcess:
    """Tests for NonErgodicGenerativeProcess class."""

    @pytest.fixture
    def two_coin_process(self):
        """Two biased coins as a nonergodic mixture."""
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})
        return NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[0.6, 0.4],
        )

    def test_vocab_size_inferred_correctly(self, two_coin_process):
        """Vocab size should be max of component vocab sizes."""
        assert two_coin_process.vocab_size == 2

    def test_initial_state_has_correct_structure(self, two_coin_process):
        """Initial state should have component beliefs and per-component states."""
        state = two_coin_process.initial_state
        assert isinstance(state, NonErgodicState)
        chex.assert_trees_all_close(state.component_beliefs, jnp.array([0.6, 0.4]))
        assert len(state.component_states) == 2

    def test_observation_distribution_is_mixture(self, two_coin_process):
        """Observation dist should be weighted mixture of component dists."""
        state = two_coin_process.initial_state
        dist = two_coin_process.observation_probability_distribution(state)

        # Expected: 0.6 * [0.7, 0.3] + 0.4 * [0.3, 0.7] = [0.54, 0.46]
        expected = jnp.array([0.54, 0.46])
        chex.assert_trees_all_close(dist, expected, atol=1e-6)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)

    def test_transition_updates_beliefs_correctly(self, two_coin_process):
        """Observing a token should update component beliefs via Bayes rule."""
        state = two_coin_process.initial_state

        # Observe token 0 (heads)
        new_state = two_coin_process.transition_states(state, jnp.array(0))

        # Bayes update: P(comp | obs=0) proportional to P(obs=0 | comp) * P(comp)
        # P(comp0 | obs=0) proportional to 0.7 * 0.6 = 0.42
        # P(comp1 | obs=0) proportional to 0.3 * 0.4 = 0.12
        # Normalized: [0.42, 0.12] / 0.54 = [0.778, 0.222]
        expected_beliefs = jnp.array([0.42, 0.12])
        expected_beliefs = expected_beliefs / jnp.sum(expected_beliefs)
        chex.assert_trees_all_close(new_state.component_beliefs, expected_beliefs, atol=1e-5)

    def test_probability_equals_mixture_probability(self, two_coin_process):
        """P(sequence) should equal weighted sum of component probabilities."""
        observations = jnp.array([0, 0, 1])  # HHT

        prob = two_coin_process.probability(observations)

        # Manual calculation:
        # P(HHT | coin1) = 0.7 * 0.7 * 0.3 = 0.147
        # P(HHT | coin2) = 0.3 * 0.3 * 0.7 = 0.063
        # P(HHT) = 0.6 * 0.147 + 0.4 * 0.063 = 0.0882 + 0.0252 = 0.1134
        expected = 0.6 * 0.147 + 0.4 * 0.063
        chex.assert_trees_all_close(prob, expected, atol=1e-6)

    def test_log_probability_consistent_with_probability(self, two_coin_process):
        """log_probability should equal log of probability."""
        observations = jnp.array([0, 1, 0, 1])

        prob = two_coin_process.probability(observations)
        log_prob = two_coin_process.log_probability(observations)

        chex.assert_trees_all_close(log_prob, jnp.log(prob), atol=1e-5)

    def test_generate_produces_valid_sequences(self, two_coin_process):
        """generate should produce sequences within vocab range."""
        state = two_coin_process.initial_state
        # Batch the state
        batch_size = 4
        batch_states = NonErgodicState(
            component_beliefs=jnp.broadcast_to(state.component_beliefs, (batch_size,) + state.component_beliefs.shape),
            component_states=tuple(jnp.broadcast_to(s, (batch_size,) + s.shape) for s in state.component_states),
        )
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        final_states, observations = two_coin_process.generate(batch_states, keys, 10, False)

        assert observations.shape == (batch_size, 10)
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < two_coin_process.vocab_size)

    def test_emit_observation_within_vocab(self, two_coin_process):
        """emit_observation should return valid tokens."""
        state = two_coin_process.initial_state
        key = jax.random.PRNGKey(42)

        obs = two_coin_process.emit_observation(state, key)

        assert obs.shape == ()
        assert 0 <= int(obs) < two_coin_process.vocab_size


class TestVocabMaps:
    """Tests for vocabulary mapping functionality."""

    def test_different_vocab_maps_work(self):
        """Components with different vocab maps should be handled correctly."""
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})

        process = NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[0.5, 0.5],
            vocab_maps=[[0, 1], [0, 2]],  # coin2 maps to tokens 0, 2
        )

        assert process.vocab_size == 3  # tokens 0, 1, 2

        state = process.initial_state
        dist = process.observation_probability_distribution(state)

        # Token 0: both components can emit (0.5 * 0.7 + 0.5 * 0.3 = 0.5)
        # Token 1: only component 0 (0.5 * 0.3 = 0.15)
        # Token 2: only component 1 (0.5 * 0.7 = 0.35)
        expected = jnp.array([0.5, 0.15, 0.35])
        chex.assert_trees_all_close(dist, expected, atol=1e-6)

    def test_unmapped_tokens_have_zero_probability(self):
        """Tokens not in a component's vocab should contribute zero from that component."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})

        process = NonErgodicGenerativeProcess(
            components=[coin],
            component_weights=[1.0],
            vocab_maps=[[0, 2]],  # Component uses tokens 0, 2; token 1 is unmapped
        )

        state = process.initial_state
        dist = process.observation_probability_distribution(state)

        assert process.vocab_size == 3
        assert dist[1] == 0.0  # Token 1 has zero probability


class TestMixedComponentTypes:
    """Tests for mixing different GenerativeProcess types."""

    def test_hmm_and_ghmm_mixture(self):
        """Should handle mixing HMM and GHMM components."""
        hmm = build_hidden_markov_model("even_ones", {"p": 0.5})
        ghmm = build_generalized_hidden_markov_model("tom_quantum", {"alpha": 1.0, "beta": 1.0})

        process = NonErgodicGenerativeProcess(
            components=[hmm, ghmm],
            component_weights=[0.7, 0.3],
        )

        state = process.initial_state
        dist = process.observation_probability_distribution(state)

        chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)
        assert jnp.all(dist >= 0)


class TestBuilder:
    """Tests for build_nonergodic_process_from_spec."""

    def test_build_from_hmm_specs(self):
        """Should build process from HMM specifications."""
        process = build_nonergodic_process_from_spec(
            components=[
                {
                    "component_type": "hmm",
                    "process_name": "coin",
                    "process_params": {"p": 0.6},
                },
                {
                    "component_type": "hmm",
                    "process_name": "coin",
                    "process_params": {"p": 0.4},
                },
            ],
            component_weights=[0.5, 0.5],
        )

        assert isinstance(process, NonErgodicGenerativeProcess)
        assert len(process.components) == 2
        assert process.vocab_size == 2

    def test_build_from_ghmm_specs(self):
        """Should build process from GHMM specifications."""
        process = build_nonergodic_process_from_spec(
            components=[
                {
                    "component_type": "ghmm",
                    "process_name": "tom_quantum",
                    "process_params": {"alpha": 1.0, "beta": 1.0},
                },
            ],
            component_weights=[1.0],
        )

        assert isinstance(process, NonErgodicGenerativeProcess)
        assert len(process.components) == 1

    def test_build_with_vocab_maps(self):
        """Should respect vocab_maps in spec."""
        process = build_nonergodic_process_from_spec(
            components=[
                {
                    "component_type": "hmm",
                    "process_name": "coin",
                    "process_params": {"p": 0.5},
                },
                {
                    "component_type": "hmm",
                    "process_name": "coin",
                    "process_params": {"p": 0.5},
                },
            ],
            component_weights=[0.5, 0.5],
            vocab_maps=[[0, 1], [0, 2]],
        )

        assert process.vocab_size == 3

    def test_invalid_component_type_raises(self):
        """Should raise for unknown component type."""
        with pytest.raises(ValueError, match="Unknown component_type"):
            build_nonergodic_process_from_spec(
                components=[{"component_type": "invalid", "process_name": "coin"}],
                component_weights=[1.0],
            )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_component_degenerates_to_component(self):
        """Single-component process should behave like the component."""
        coin = build_hidden_markov_model("coin", {"p": 0.7})

        process = NonErgodicGenerativeProcess(
            components=[coin],
            component_weights=[1.0],
        )

        observations = jnp.array([0, 1, 0])

        process_prob = process.probability(observations)
        coin_prob = coin.probability(observations)

        chex.assert_trees_all_close(process_prob, coin_prob, atol=1e-6)

    def test_weights_are_normalized(self):
        """Component weights should be normalized to sum to 1."""
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})

        # Provide unnormalized weights
        process = NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[2.0, 3.0],  # Sum to 5, not 1
        )

        chex.assert_trees_all_close(process.component_weights, jnp.array([0.4, 0.6]), atol=1e-6)

    def test_empty_components_raises(self):
        """Should raise for empty component list."""
        with pytest.raises(ValueError, match="at least one component"):
            NonErgodicGenerativeProcess(
                components=[],
                component_weights=[],
            )

    def test_mismatched_weights_raises(self):
        """Should raise if weights don't match component count."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})

        with pytest.raises(ValueError, match="must match"):
            NonErgodicGenerativeProcess(
                components=[coin, coin],
                component_weights=[1.0],  # Only 1 weight for 2 components
            )

    def test_mismatched_vocab_maps_raises(self):
        """Should raise if vocab map count doesn't match component count."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})

        with pytest.raises(ValueError, match="Length of vocab maps"):
            NonErgodicGenerativeProcess(
                components=[coin, coin],
                component_weights=[0.5, 0.5],
                vocab_maps=[[0, 1]],
            )

    def test_duplicate_vocab_map_entries_raise(self):
        """Should raise if a component vocab map reuses a global token index."""
        coin = build_hidden_markov_model("coin", {"p": 0.5})

        with pytest.raises(ValueError, match="must not contain duplicate"):
            NonErgodicGenerativeProcess(
                components=[coin],
                component_weights=[1.0],
                vocab_maps=[[0, 0]],
            )


class TestGenerateReturnAllStates:
    """Tests for generate with return_all_states=True."""

    @pytest.fixture
    def two_mess3_process(self):
        """Two mess3 HMMs as a nonergodic mixture."""
        hmm1 = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
        hmm2 = build_hidden_markov_model("mess3", {"x": 0.5, "a": 0.6})
        return NonErgodicGenerativeProcess(
            components=[hmm1, hmm2],
            component_weights=[0.6, 0.4],
        )

    def test_return_all_states_shapes(self, two_mess3_process):
        """Both component_beliefs and component_states should have time dimension."""
        batch_size = 4
        seq_len = 8
        state = two_mess3_process.initial_state
        batch_states = NonErgodicState(
            component_beliefs=jnp.broadcast_to(state.component_beliefs, (batch_size,) + state.component_beliefs.shape),
            component_states=tuple(jnp.broadcast_to(s, (batch_size,) + s.shape) for s in state.component_states),
        )
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        trajectory, observations = two_mess3_process.generate(batch_states, keys, seq_len, True)

        assert observations.shape == (batch_size, seq_len)
        assert trajectory.component_beliefs.shape == (batch_size, seq_len, 2)
        for i, comp in enumerate(two_mess3_process.components):
            assert trajectory.component_states[i].shape == (batch_size, seq_len, comp.initial_state.shape[0])

    def test_return_all_states_beliefs_are_valid_distributions(self, two_mess3_process):
        """Component beliefs at each timestep should sum to 1."""
        batch_size = 4
        seq_len = 8
        state = two_mess3_process.initial_state
        batch_states = NonErgodicState(
            component_beliefs=jnp.broadcast_to(state.component_beliefs, (batch_size,) + state.component_beliefs.shape),
            component_states=tuple(jnp.broadcast_to(s, (batch_size,) + s.shape) for s in state.component_states),
        )
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        trajectory, _ = two_mess3_process.generate(batch_states, keys, seq_len, True)

        belief_sums = jnp.sum(trajectory.component_beliefs, axis=-1)
        chex.assert_trees_all_close(belief_sums, jnp.ones_like(belief_sums), atol=1e-5)


class TestFactoredComponent:
    """Tests for FactoredGenerativeProcess as a NonErgodic component."""

    @pytest.fixture
    def hmm_factored_process(self):
        """NonErgodic process with one HMM and one factored component."""
        hmm = build_hidden_markov_model("coin", {"p": 0.7})
        factored = build_factored_process_from_spec(
            structure_type="independent",
            spec=[
                {"component_type": "hmm", "variants": [{"process_name": "coin", "process_params": {"p": 0.6}}]},
                {"component_type": "hmm", "variants": [{"process_name": "coin", "process_params": {"p": 0.4}}]},
            ],
        )
        return NonErgodicGenerativeProcess(
            components=[hmm, factored],
            component_weights=[0.5, 0.5],
        )

    def test_factored_component_generate(self, hmm_factored_process):
        """NonErgodic with a factored component should generate valid sequences."""
        process = hmm_factored_process
        batch_size = 4
        seq_len = 6
        batch_states = _expand_state(process.initial_state, batch_size)
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        final_states, observations = process.generate(batch_states, keys, seq_len, False)

        assert observations.shape == (batch_size, seq_len)
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < process.vocab_size)

    def test_factored_component_return_all_states(self, hmm_factored_process):
        """Factored component state trajectory should have correct shapes."""
        process = hmm_factored_process

        batch_size = 4
        seq_len = 6
        batch_states = _expand_state(process.initial_state, batch_size)
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        trajectory, observations = process.generate(batch_states, keys, seq_len, True)

        assert observations.shape == (batch_size, seq_len)
        assert trajectory.component_beliefs.shape == (batch_size, seq_len, 2)
        # HMM component state: flat array
        assert trajectory.component_states[0].ndim == 3  # [batch, seq, state_dim]
        # Factored component state: tuple of arrays
        assert isinstance(trajectory.component_states[1], tuple)
        for factor_state in trajectory.component_states[1]:
            assert factor_state.ndim == 3  # [batch, seq, factor_dim]


class TestGenerateDataBatchWithFullHistory:
    """Tests for generate_data_batch_with_full_history with NonErgodicGenerativeProcess."""

    def test_full_history_shapes(self):
        """Belief states should have consistent shapes after slicing."""
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})
        process = NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[0.6, 0.4],
        )

        batch_size = 4
        seq_len = 8
        batch_states = _expand_state(process.initial_state, batch_size)

        result = generate_data_batch_with_full_history(
            batch_states,  # type: ignore[arg-type]
            process,
            batch_size,
            seq_len,
            jax.random.PRNGKey(0),
        )

        belief_states = result["belief_states"]
        inputs = result["inputs"]
        assert isinstance(inputs, jax.Array)

        assert isinstance(belief_states, NonErgodicState)
        input_len = inputs.shape[1]
        assert belief_states.component_beliefs.shape == (batch_size, input_len, 2)
        for cs in belief_states.component_states:
            assert not isinstance(cs, tuple)
            assert cs.shape[0] == batch_size
            assert cs.shape[1] == input_len

    def test_full_history_with_bos(self):
        """Belief states should align with inputs when BOS token is used."""
        coin1 = build_hidden_markov_model("coin", {"p": 0.7})
        coin2 = build_hidden_markov_model("coin", {"p": 0.3})
        process = NonErgodicGenerativeProcess(
            components=[coin1, coin2],
            component_weights=[0.6, 0.4],
        )

        batch_size = 4
        seq_len = 8
        bos_token = process.vocab_size
        batch_states = _expand_state(process.initial_state, batch_size)

        result = generate_data_batch_with_full_history(
            batch_states,  # type: ignore[arg-type]
            process,
            batch_size,
            seq_len,
            jax.random.PRNGKey(0),
            bos_token=bos_token,
        )

        belief_states = result["belief_states"]
        inputs = result["inputs"]
        assert isinstance(inputs, jax.Array)

        assert isinstance(belief_states, NonErgodicState)
        input_len = inputs.shape[1]
        assert belief_states.component_beliefs.shape == (batch_size, input_len, 2)
        for cs in belief_states.component_states:
            assert not isinstance(cs, tuple)
            assert cs.shape[0] == batch_size
            assert cs.shape[1] == input_len
