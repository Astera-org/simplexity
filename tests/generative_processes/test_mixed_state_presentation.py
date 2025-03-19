import math
from typing import cast

import chex
import jax.numpy as jnp
import pytest

from simplexity.data_structures import Collection, Queue, Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import (
    LogMixedStateNode,
    LogMixedStateTree,
    LogMixedStateTreeGenerator,
    LogNodeDictValue,
    MixedStateNode,
    MixedStateTree,
    MixedStateTreeGenerator,
    NodeDictValue,
    SearchAlgorithm,
    Sequence,
)
from simplexity.generative_processes.transition_matrices import no_consecutive_ones

ABS_TOL = 1e-7

NODES = {
    "": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(0, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([2 / 3, 1 / 3]),
        belief_state=jnp.array([2 / 3, 1 / 3]),
        probability=jnp.array(1),
    ),
    "0": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([2 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(2 / 3),
    ),
    "1": MixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 1 / 3]),
        belief_state=jnp.array([0, 1]),
        probability=jnp.array(1 / 3),
    ),
    "00": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([1 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(1 / 3),
    ),
    "01": MixedStateNode(
        sequence=jnp.array([0, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 1 / 3]),
        belief_state=jnp.array([0, 1]),
        probability=jnp.array(1 / 3),
    ),
    "10": MixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([1 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(1 / 3),
    ),
    "11": MixedStateNode(
        sequence=jnp.array([1, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 0]),
        belief_state=jnp.array([jnp.nan, jnp.nan]),
        probability=jnp.array(0),
    ),
}

LOG_NODES = {
    "": LogMixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(0, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([2 / 3, 1 / 3]),
        belief_state=jnp.array([2 / 3, 1 / 3]),
        probability=jnp.array(1),
        log_unnormalized_belief_state=jnp.log(jnp.array([2 / 3, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([2 / 3, 1 / 3])),
        log_probability=jnp.log(1),
    ),
    "0": LogMixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([2 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(2 / 3),
        log_unnormalized_belief_state=jnp.log(jnp.array([2 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(2 / 3),
    ),
    "1": LogMixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 1 / 3]),
        belief_state=jnp.array([0, 1]),
        probability=jnp.array(1 / 3),
        log_unnormalized_belief_state=jnp.log(jnp.array([0, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([0, 1])),
        log_probability=jnp.log(1 / 3),
    ),
    "00": LogMixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([1 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(1 / 3),
        log_unnormalized_belief_state=jnp.log(jnp.array([1 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(1 / 3),
    ),
    "01": LogMixedStateNode(
        sequence=jnp.array([0, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 1 / 3]),
        belief_state=jnp.array([0, 1]),
        probability=jnp.array(1 / 3),
        log_unnormalized_belief_state=jnp.log(jnp.array([0, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([0, 1])),
        log_probability=jnp.log(1 / 3),
    ),
    "10": LogMixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([1 / 3, 0]),
        belief_state=jnp.array([1, 0]),
        probability=jnp.array(1 / 3),
        log_unnormalized_belief_state=jnp.log(jnp.array([1 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(1 / 3),
    ),
    "11": LogMixedStateNode(
        sequence=jnp.array([1, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        unnormalized_belief_state=jnp.array([0, 0]),
        belief_state=jnp.array([jnp.nan, jnp.nan]),
        probability=jnp.array(0),
        log_unnormalized_belief_state=jnp.log(jnp.array([0, 0])),
        log_belief_state=jnp.log(jnp.array([jnp.nan, jnp.nan])),
        log_probability=jnp.log(0),
    ),
}


@pytest.fixture
def generator() -> MixedStateTreeGenerator:
    transition_matrices = no_consecutive_ones()
    ghmm = GeneralizedHiddenMarkovModel(transition_matrices)
    return MixedStateTreeGenerator(ghmm, max_sequence_length=2)


@pytest.fixture
def log_generator() -> LogMixedStateTreeGenerator:
    transition_matrices = no_consecutive_ones()
    hmm = HiddenMarkovModel(transition_matrices)
    return LogMixedStateTreeGenerator(hmm, max_sequence_length=2, prob_threshold=-jnp.inf)


def get_sequences_in_collection(collection: Queue[MixedStateNode]) -> list[tuple[int, ...]]:
    nodes = cast(MixedStateNode, collection.data)
    sequences = []
    for i in range(collection.size):
        sequence = tuple(nodes.sequence[i][: nodes.sequence_length[i]].tolist())
        sequences.append(sequence)
    return sequences


@pytest.mark.parametrize(("generator_name", "expected_nodes"), [("generator", NODES), ("log_generator", LOG_NODES)])
def test_get_child(generator_name: str, expected_nodes: dict[str, MixedStateNode], request: pytest.FixtureRequest):
    generator = cast(MixedStateTreeGenerator, request.getfixturevalue(generator_name))
    child = generator.get_child(generator.root, jnp.array(0))
    chex.assert_trees_all_close(child, expected_nodes["0"], atol=ABS_TOL)

    child = generator.get_child(child, jnp.array(1))
    chex.assert_trees_all_close(child, expected_nodes["01"])


@pytest.mark.parametrize("generator_name", ["generator", "log_generator"])
def test_get_all_children(generator_name: str, request: pytest.FixtureRequest):
    generator = cast(MixedStateTreeGenerator, request.getfixturevalue(generator_name))
    search_nodes = Queue(max_size=7, default_element=generator.root)
    search_nodes = search_nodes.enqueue(generator.root)
    sequences = get_sequences_in_collection(search_nodes)
    assert set(sequences) == {()}
    search_nodes = generator.get_all_children(search_nodes)
    assert search_nodes.size == 2
    sequences = get_sequences_in_collection(search_nodes)
    assert set(sequences) == {(0,), (1,)}
    search_nodes = generator.get_all_children(search_nodes)
    assert search_nodes.size == 4
    sequences = get_sequences_in_collection(search_nodes)
    assert set(sequences) == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize(("generator_name", "expected_nodes"), [("generator", NODES), ("log_generator", LOG_NODES)])
@pytest.mark.parametrize(
    ("data_structure", "expected_sequences"),
    [(Stack, ["", "1", "11", "10", "0", "01", "00"]), (Queue, ["", "0", "1", "00", "01", "10", "11"])],
)
def test_next_node(
    generator_name: str,
    expected_nodes: dict[str, MixedStateNode],
    data_structure: type[Collection[MixedStateNode]],
    expected_sequences: list[str],
    request: pytest.FixtureRequest,
):
    generator = cast(MixedStateTreeGenerator, request.getfixturevalue(generator_name))
    search_nodes = data_structure(max_size=7, default_element=generator.root)
    search_nodes = search_nodes.add(generator.root)

    for expected_sequence in expected_sequences:
        search_nodes, node = generator._next_node(search_nodes)
        expected_node = expected_nodes[expected_sequence]
        chex.assert_trees_all_close(node, expected_node, atol=ABS_TOL)

    assert search_nodes.is_empty


@pytest.mark.parametrize("search_algorithm", [SearchAlgorithm.BREADTH_FIRST, SearchAlgorithm.DEPTH_FIRST])
def test_generate(generator: MixedStateTreeGenerator, search_algorithm: SearchAlgorithm):
    tree = generator.generate(search_algorithm)
    assert isinstance(tree, MixedStateTree)
    expected_nodes: dict[Sequence, NodeDictValue] = {
        (): NodeDictValue(probability=1, belief_state=(2 / 3, 1 / 3)),
        (0,): NodeDictValue(probability=2 / 3, belief_state=(1, 0)),
        (1,): NodeDictValue(probability=1 / 3, belief_state=(0, 1)),
        (0, 0): NodeDictValue(probability=1 / 3, belief_state=(1, 0)),
        (0, 1): NodeDictValue(probability=1 / 3, belief_state=(0, 1)),
        (1, 0): NodeDictValue(probability=1 / 3, belief_state=(1, 0)),
        (1, 1): NodeDictValue(probability=0, belief_state=(jnp.nan, jnp.nan)),
    }
    assert set(tree.nodes.keys()) == set(expected_nodes.keys())

    def assert_node_dict_values_close(actual: NodeDictValue, expected: NodeDictValue):
        assert math.isclose(actual[0], expected[0], abs_tol=1e-6)
        for actual_state_prob, expected_state_prob in zip(actual[1], expected[1], strict=True):
            if math.isnan(expected_state_prob):
                assert math.isnan(actual_state_prob)
            else:
                assert math.isclose(actual_state_prob, expected_state_prob, abs_tol=1e-6)

    for sequence in tree.nodes:
        node = tree.nodes[sequence]
        assert isinstance(node, NodeDictValue)
        assert_node_dict_values_close(node, expected_nodes[sequence])


@pytest.mark.parametrize("search_algorithm", [SearchAlgorithm.BREADTH_FIRST, SearchAlgorithm.DEPTH_FIRST])
def test_log_generate(log_generator: LogMixedStateTreeGenerator, search_algorithm: SearchAlgorithm):
    tree = log_generator.generate(search_algorithm)
    assert isinstance(tree, LogMixedStateTree)
    log_1 = 0.0
    log_2_3 = math.log(2 / 3)
    log_1_3 = math.log(1 / 3)
    log_0 = -math.inf
    expected_nodes: dict[Sequence, LogNodeDictValue] = {
        (): LogNodeDictValue(log_probability=log_1, log_belief_state=(log_2_3, log_1_3)),
        (0,): LogNodeDictValue(log_probability=log_2_3, log_belief_state=(log_1, log_0)),
        (1,): LogNodeDictValue(log_probability=log_1_3, log_belief_state=(log_0, log_1)),
        (0, 0): LogNodeDictValue(log_probability=log_1_3, log_belief_state=(log_1, log_0)),
        (0, 1): LogNodeDictValue(log_probability=log_1_3, log_belief_state=(log_0, log_1)),
        (1, 0): LogNodeDictValue(log_probability=log_1_3, log_belief_state=(log_1, log_0)),
        (1, 1): LogNodeDictValue(log_probability=log_0, log_belief_state=(math.nan, math.nan)),
    }
    assert set(tree.nodes.keys()) == set(expected_nodes.keys())

    def assert_node_dict_values_close(actual: LogNodeDictValue, expected: LogNodeDictValue):
        assert math.isclose(actual[0], expected[0], abs_tol=ABS_TOL)
        for actual_state_log_prob, expected_state_log_prob in zip(actual[1], expected[1], strict=True):
            if math.isnan(expected_state_log_prob):
                assert math.isnan(actual_state_log_prob)
            else:
                assert math.isclose(actual_state_log_prob, expected_state_log_prob, abs_tol=ABS_TOL)

    for sequence in tree.nodes:
        node = tree.nodes[sequence]
        assert isinstance(node, LogNodeDictValue)
        assert_node_dict_values_close(node, expected_nodes[sequence])


@pytest.mark.parametrize("generator_name", ["generator", "log_generator"])
def test_myopic_entropy(generator_name: str, request: pytest.FixtureRequest):
    generator = cast(MixedStateTreeGenerator, request.getfixturevalue(generator_name))
    myopic_entropies = generator.compute_myopic_entropy()
    assert myopic_entropies.sequence_lengths.shape == (generator.max_sequence_length + 1,)
    assert myopic_entropies.belief_state_entropies.shape == (generator.max_sequence_length + 1,)
    assert jnp.all(~jnp.isnan(myopic_entropies.belief_state_entropies))
    assert jnp.all(
        myopic_entropies.belief_state_entropies[1:] - myopic_entropies.belief_state_entropies[:-1] <= 0 + ABS_TOL
    ), "Belief state myopic entropy should be monotonically non-increasing with sequence length"
    assert myopic_entropies.observation_entropies.shape == (generator.max_sequence_length + 1,)
    assert jnp.all(~jnp.isnan(myopic_entropies.observation_entropies))
    assert jnp.all(
        myopic_entropies.observation_entropies[1:] - myopic_entropies.observation_entropies[:-1] <= 0 + ABS_TOL
    ), "Observation myopic entropy should be monotonically non-increasing with sequence length"
