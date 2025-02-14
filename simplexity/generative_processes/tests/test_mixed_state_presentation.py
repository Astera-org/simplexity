import math

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.data_structures import Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import MixedStateNode, MixedStateTreeGenerator, NodeDict
from simplexity.generative_processes.transition_matrices import no_consecutive_ones

NODE_0 = MixedStateNode(
    sequence=jnp.array([0, 0], dtype=jnp.int32),
    sequence_length=jnp.array(1, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([2 / 3, 0])),
)

NODE_1 = MixedStateNode(
    sequence=jnp.array([1, 0], dtype=jnp.int32),
    sequence_length=jnp.array(1, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([0, 1 / 3])),
)

NODE_00 = MixedStateNode(
    sequence=jnp.array([0, 0], dtype=jnp.int32),
    sequence_length=jnp.array(2, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([1 / 3, 0])),
)

NODE_01 = MixedStateNode(
    sequence=jnp.array([0, 1], dtype=jnp.int32),
    sequence_length=jnp.array(2, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([0, 1 / 3])),
)

NODE_10 = MixedStateNode(
    sequence=jnp.array([1, 0], dtype=jnp.int32),
    sequence_length=jnp.array(2, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([1 / 3, 0])),
)

NODE_11 = MixedStateNode(
    sequence=jnp.array([1, 1], dtype=jnp.int32),
    sequence_length=jnp.array(2, dtype=jnp.int32),
    log_state_distribution=jnp.log(jnp.array([0, 0])),
)


@pytest.fixture
def generator() -> MixedStateTreeGenerator:
    transition_matrices = no_consecutive_ones()
    ghmm = GeneralizedHiddenMarkovModel(transition_matrices)
    return MixedStateTreeGenerator(ghmm, max_sequence_length=2, max_tree_size=7, max_stack_size=7)


def test_get_child(generator: MixedStateTreeGenerator):
    child = generator.get_child(generator.root, jnp.array(0))
    chex.assert_trees_all_close(child, NODE_0)
    chex.assert_trees_all_close(child.log_probability, jnp.log(2 / 3))

    child = generator.get_child(child, jnp.array(1))
    chex.assert_trees_all_close(child, NODE_01)
    chex.assert_trees_all_close(child.log_probability, jnp.log(1 / 3))


def test_next_node(generator: MixedStateTreeGenerator):
    stack = Stack(max_size=7, default_element=generator.root)
    stack = stack.push(generator.root)

    expected_nodes = [generator.root, NODE_1, NODE_11, NODE_10, NODE_0, NODE_01, NODE_00]
    for expected_node in expected_nodes:
        stack, node = generator._next_node(stack)
        chex.assert_trees_all_close(node, expected_node)

    assert stack.is_empty


def test_generate(generator: MixedStateTreeGenerator):
    tree = generator.generate()
    expected_nodes: NodeDict = {
        (): math.log(1),
        (0,): math.log(2 / 3),
        (1,): math.log(1 / 3),
        (0, 0): math.log(1 / 3),
        (0, 1): math.log(1 / 3),
        (1, 0): math.log(1 / 3),
        (1, 1): -math.inf,
    }
    assert set(tree.nodes.keys()) == set(expected_nodes.keys())
    for sequence, log_prob in tree.nodes.items():
        assert math.isclose(log_prob, expected_nodes[sequence], abs_tol=1e-7)
