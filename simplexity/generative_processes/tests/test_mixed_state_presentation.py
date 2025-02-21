import math

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.data_structures import Collection, Queue, Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import (
    MixedStateNode,
    MixedStateTreeGenerator,
    NodeDict,
    NodeDictValue,
    SearchAlgorithm,
)
from simplexity.generative_processes.transition_matrices import no_consecutive_ones

NODES = {
    "": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(0, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([2 / 3, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([2 / 3, 1 / 3])),
        log_probability=jnp.log(1),
    ),
    "0": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([2 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(2 / 3),
    ),
    "1": MixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(1, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([0, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([0, 1])),
        log_probability=jnp.log(1 / 3),
    ),
    "00": MixedStateNode(
        sequence=jnp.array([0, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([1 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(1 / 3),
    ),
    "01": MixedStateNode(
        sequence=jnp.array([0, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([0, 1 / 3])),
        log_belief_state=jnp.log(jnp.array([0, 1])),
        log_probability=jnp.log(1 / 3),
    ),
    "10": MixedStateNode(
        sequence=jnp.array([1, 0], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([1 / 3, 0])),
        log_belief_state=jnp.log(jnp.array([1, 0])),
        log_probability=jnp.log(1 / 3),
    ),
    "11": MixedStateNode(
        sequence=jnp.array([1, 1], dtype=jnp.int32),
        sequence_length=jnp.array(2, dtype=jnp.int32),
        log_state=jnp.log(jnp.array([0, 0])),
        log_belief_state=jnp.log(jnp.array([jnp.nan, jnp.nan])),  # TODO: fix normalization
        log_probability=jnp.log(0),
    ),
}


@pytest.fixture
def generator() -> MixedStateTreeGenerator:
    transition_matrices = no_consecutive_ones()
    ghmm = GeneralizedHiddenMarkovModel(transition_matrices)
    return MixedStateTreeGenerator(ghmm, max_sequence_length=2)


def test_get_child(generator: MixedStateTreeGenerator):
    child = generator.get_child(generator.root, jnp.array(0))
    chex.assert_trees_all_close(child, NODES["0"])
    chex.assert_trees_all_close(child.log_probability, jnp.log(2 / 3))

    child = generator.get_child(child, jnp.array(1))
    chex.assert_trees_all_close(child, NODES["01"])
    chex.assert_trees_all_close(child.log_probability, jnp.log(1 / 3))


@pytest.mark.parametrize(
    ("data_structure", "expected_sequences"),
    [(Stack, ["", "1", "11", "10", "0", "01", "00"]), (Queue, ["", "0", "1", "00", "01", "10", "11"])],
)
def test_next_node(
    generator: MixedStateTreeGenerator, data_structure: type[Collection[MixedStateNode]], expected_sequences: list[str]
):
    search_nodes = data_structure(max_size=7, default_element=generator.root)
    search_nodes = search_nodes.add(generator.root)

    for expected_sequence in expected_sequences:
        search_nodes, node = generator._next_node(search_nodes)
        expected_node = NODES[expected_sequence]
        chex.assert_trees_all_close(node, expected_node)

    assert search_nodes.is_empty


@pytest.mark.parametrize("search_algorithm", [SearchAlgorithm.BREADTH_FIRST, SearchAlgorithm.DEPTH_FIRST])
def test_generate(generator: MixedStateTreeGenerator, search_algorithm: SearchAlgorithm):
    tree = generator.generate(search_algorithm)
    log_1 = 0.0
    log_2_3 = math.log(2 / 3)
    log_1_3 = math.log(1 / 3)
    log_0 = -math.inf
    expected_nodes: NodeDict = {
        (): NodeDictValue(log_probability=log_1, log_belief_state=(log_2_3, log_1_3)),
        (0,): NodeDictValue(log_probability=log_2_3, log_belief_state=(log_1, log_0)),
        (1,): NodeDictValue(log_probability=log_1_3, log_belief_state=(log_0, log_1)),
        (0, 0): NodeDictValue(log_probability=log_1_3, log_belief_state=(log_1, log_0)),
        (0, 1): NodeDictValue(log_probability=log_1_3, log_belief_state=(log_0, log_1)),
        (1, 0): NodeDictValue(log_probability=log_1_3, log_belief_state=(log_1, log_0)),
        (1, 1): NodeDictValue(log_probability=log_0, log_belief_state=(math.nan, math.nan)),  # TODO: fix normalization
    }
    assert set(tree.nodes.keys()) == set(expected_nodes.keys())

    def assert_node_dict_values_close(actual: NodeDictValue, expected: NodeDictValue):
        assert math.isclose(actual[0], expected[0], abs_tol=1e-7)
        for actual_state_log_prob, expected_state_log_prob in zip(actual[1], expected[1], strict=True):
            if math.isnan(expected_state_log_prob):
                assert math.isnan(actual_state_log_prob)
            else:
                assert math.isclose(actual_state_log_prob, expected_state_log_prob, abs_tol=1e-7)

    for sequence in tree.nodes:
        assert_node_dict_values_close(tree.nodes[sequence], expected_nodes[sequence])
