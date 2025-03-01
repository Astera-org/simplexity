from enum import Enum
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.data_structures import Collection, Queue, Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.log_math import entropy

Sequence = tuple[int, ...]
LogProbability = float
LogBeliefState = tuple[float, ...]


class NodeDictValue(NamedTuple):
    """The value of a node in the node dictionary."""

    log_probability: LogProbability
    log_belief_state: LogBeliefState


NodeDict = dict[Sequence, NodeDictValue]


class MixedStateNode(eqx.Module):
    """A node in a mixed state presentation of a generative process."""

    sequence: jax.Array
    sequence_length: jax.Array
    log_state: jax.Array
    log_belief_state: jax.Array
    log_probability: jax.Array

    @property
    def num_states(self) -> int:
        """The number of states in the node."""
        return self.log_belief_state.shape[0]

    @property
    def max_sequence_length(self) -> int:
        """The maximum length of the sequence."""
        return self.sequence.shape[0]


class TreeData(eqx.Module):
    """Data for a tree."""

    sequences: jax.Array
    sequence_lengths: jax.Array
    log_belief_states: jax.Array
    log_probabilities: jax.Array
    size: jax.Array

    @classmethod
    def empty(cls, max_size: int, max_sequence_length: int, num_states: int) -> "TreeData":
        """Create an empty tree."""
        return cls(
            sequences=jnp.zeros((max_size, max_sequence_length), dtype=jnp.int32),
            sequence_lengths=jnp.zeros((max_size,), dtype=jnp.int32),
            log_belief_states=jnp.zeros((max_size, num_states), dtype=jnp.float32),
            log_probabilities=jnp.zeros((max_size,), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def add(self, node: MixedStateNode) -> "TreeData":
        """Add a sequence to the tree."""
        return TreeData(
            sequences=self.sequences.at[self.size].set(node.sequence),
            sequence_lengths=self.sequence_lengths.at[self.size].set(node.sequence_length),
            log_belief_states=self.log_belief_states.at[self.size].set(node.log_belief_state),
            log_probabilities=self.log_probabilities.at[self.size].set(node.log_probability),
            size=self.size + 1,
        )

    @property
    def max_size(self) -> int:
        """The maximum number of elements in the tree."""
        return self.sequences.shape[0]

    @property
    def max_sequence_length(self) -> int:
        """The maximum length of the sequences."""
        return self.sequences.shape[1]

    @property
    def num_states(self) -> int:
        """The number of states in the tree."""
        return self.log_belief_states.shape[1]


class MixedStateTree:
    """A presentation of a generative process as a mixed state."""

    def __init__(self, nodes: TreeData):
        self.nodes: NodeDict = {}
        for i in range(nodes.size):
            self.add(
                nodes.sequences[i, : nodes.sequence_lengths[i]],
                nodes.log_probabilities[i],
                nodes.log_belief_states[i],
            )

    def __len__(self) -> int:
        """The number of nodes in the tree."""
        return len(self.nodes)

    def add(self, sequence: jax.Array, log_probability: jax.Array, log_belief_state: jax.Array) -> None:
        """Add a sequence to the tree."""
        sequence_ = tuple(sequence.tolist())
        log_probability_ = log_probability.item()
        log_belief_state_ = tuple(log_belief_state.tolist())
        self.nodes[sequence_] = NodeDictValue(log_probability_, log_belief_state_)


class SearchAlgorithm(Enum):
    """The algorithm to use for searching the tree."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"


class MyopicEntropies(eqx.Module):
    """The myopic entropies of a generative process."""

    belief_state_entropies: jax.Array
    observation_entropies: jax.Array
    sequence_lengths: jax.Array

    def __init__(self, belief_state_entropies: jax.Array, observation_entropies: jax.Array):
        assert belief_state_entropies.shape == observation_entropies.shape
        self.belief_state_entropies = belief_state_entropies
        self.observation_entropies = observation_entropies
        self.sequence_lengths = jnp.arange(belief_state_entropies.shape[0])


def compute_average_entropy(log_dists: jax.Array, log_probs: jax.Array) -> jax.Array:
    """Compute the weighted average entropy of a collection of distributions."""
    entropies = eqx.filter_vmap(entropy)(log_dists)
    return jnp.sum(entropies * jnp.exp(log_probs))


class MixedStateTreeGenerator(eqx.Module):
    """A generator of nodes in a mixed state presentation of a generative process."""

    ghmm: GeneralizedHiddenMarkovModel
    max_sequence_length: int
    max_tree_size: int
    max_search_nodes_size: int
    log_prob_threshold: float

    def __init__(
        self,
        ghmm: GeneralizedHiddenMarkovModel,
        max_sequence_length: int,
        max_tree_size: int = -1,
        max_search_nodes_size: int = -1,
        log_prob_threshold: float = -jnp.inf,
    ):
        self.ghmm = ghmm
        self.max_sequence_length = max_sequence_length
        self.max_tree_size = max_tree_size
        self.max_search_nodes_size = max_search_nodes_size
        self.log_prob_threshold = log_prob_threshold

    def generate(self, search_algorithm: SearchAlgorithm = SearchAlgorithm.DEPTH_FIRST) -> MixedStateTree:
        """Generate all nodes in the tree."""

        def continue_loop(carry: tuple[TreeData, Collection[MixedStateNode]]) -> jax.Array:
            tree_data, search_nodes = carry
            return jnp.logical_and(~search_nodes.is_empty, tree_data.size < tree_data.max_size)

        def add_next_node(
            carry: tuple[TreeData, Collection[MixedStateNode]],
        ) -> tuple[TreeData, Collection[MixedStateNode]]:
            tree_data, search_nodes = carry
            search_nodes, node = self._next_node(search_nodes)
            tree_data = tree_data.add(node)
            return tree_data, search_nodes

        if self.max_tree_size < 0:
            max_tree_size = int(jnp.sum(self.ghmm.num_observations ** jnp.arange(self.max_sequence_length + 1)))
        else:
            max_tree_size = self.max_tree_size
        tree_data = TreeData.empty(max_tree_size, self.max_sequence_length, self.ghmm.num_states)

        if search_algorithm == SearchAlgorithm.BREADTH_FIRST:
            if self.max_search_nodes_size < 0:
                max_size = self.ghmm.num_observations ** (self.max_sequence_length + 1)
            else:
                max_size = self.max_search_nodes_size
            search_nodes = Queue(max_size, default_element=self.root)
        else:  # DEPTH_FIRST
            if self.max_search_nodes_size < 0:
                max_size = (self.ghmm.num_observations - 1) * self.max_sequence_length + 1
            else:
                max_size = self.max_search_nodes_size
            search_nodes = Stack(max_size, default_element=self.root)

        search_nodes = search_nodes.add(self.root)

        tree_data, _ = jax.lax.while_loop(continue_loop, add_next_node, (tree_data, search_nodes))
        return MixedStateTree(tree_data)

    def compute_myopic_entropy(self) -> MyopicEntropies:
        """Compute the myopic entropy of the generative process."""
        log_obs_dist_fn = eqx.filter_vmap(self.ghmm.log_observation_probability_distribution)

        def update_myopic_entropies(
            sequence_length: int, carry: tuple[jax.Array, jax.Array, Queue[MixedStateNode]]
        ) -> tuple[jax.Array, jax.Array, Queue[MixedStateNode]]:
            belief_state_entropies, observation_entropies, search_nodes = carry
            data = cast(MixedStateNode, search_nodes.data)
            log_obs_dists = log_obs_dist_fn(data.log_belief_state)
            belief_state_entropy = compute_average_entropy(data.log_belief_state, data.log_probability)
            observation_entropy = compute_average_entropy(log_obs_dists, data.log_probability)
            belief_state_entropies = belief_state_entropies.at[sequence_length].set(belief_state_entropy)
            observation_entropies = observation_entropies.at[sequence_length].set(observation_entropy)
            search_nodes = self.get_all_children(search_nodes)
            return belief_state_entropies, observation_entropies, search_nodes

        max_size = self.ghmm.num_observations ** (self.max_sequence_length + 1)
        if self.max_search_nodes_size > 0 and self.max_search_nodes_size < max_size:
            raise ValueError(
                f"max_search_nodes_size ({self.max_search_nodes_size}) not large enough for computing myopic entropy "
                f"up to a sequence length of {self.max_sequence_length}, a size of {max_size} is required."
            )
        search_nodes = Queue(max_size, default_element=self.root)
        search_nodes = search_nodes.add(self.root)

        belief_state_entropies = jnp.zeros(self.max_sequence_length + 1)
        observation_entropies = jnp.zeros(self.max_sequence_length + 1)
        belief_state_entropies, observation_entropies, _ = jax.lax.fori_loop(
            0,
            self.max_sequence_length + 1,
            update_myopic_entropies,
            (belief_state_entropies, observation_entropies, search_nodes),
        )
        return MyopicEntropies(belief_state_entropies, observation_entropies)

    @property
    def root(self) -> MixedStateNode:
        """The root node of the tree."""
        empty_sequence = jnp.zeros((self.max_sequence_length,), dtype=jnp.int32)
        sequence_length = jnp.array(0)
        log_state = self.ghmm.log_state_eigenvector
        log_belief_state = self.ghmm.normalize_log_belief_state(log_state)
        log_probability = jax.nn.logsumexp(log_state + self.ghmm.log_normalizing_eigenvector)
        return MixedStateNode(empty_sequence, sequence_length, log_state, log_belief_state, log_probability)

    @eqx.filter_jit
    def get_child(self, node: MixedStateNode, obs: jax.Array) -> MixedStateNode:
        """Get the child of a node."""
        sequence = node.sequence.at[node.sequence_length].set(obs)
        sequence_length = node.sequence_length + 1
        log_state = jax.nn.logsumexp(node.log_state[:, None] + self.ghmm.log_transition_matrices[obs], axis=0)
        log_belief_state = self.ghmm.normalize_log_belief_state(log_state)
        log_probability = jax.nn.logsumexp(log_state + self.ghmm.log_normalizing_eigenvector)
        return MixedStateNode(sequence, sequence_length, log_state, log_belief_state, log_probability)

    @eqx.filter_jit
    def _next_node(self, nodes: Collection[MixedStateNode]) -> tuple[Collection[MixedStateNode], MixedStateNode]:
        """Get the next node from a collection and add that node's children to the collection."""

        def add_children(
            nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
        ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
            def maybe_add_child(
                i: int, nodes_node: tuple[Collection[MixedStateNode], MixedStateNode]
            ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
                nodes, node = nodes_node
                obs = jnp.array(i)
                child = self.get_child(node, obs)

                def add_child(
                    nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
                ) -> Collection[MixedStateNode]:
                    nodes, node = nodes_node
                    return nodes.add(node)

                def do_nothing(
                    nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
                ) -> Collection[MixedStateNode]:
                    nodes, _ = nodes_node
                    return nodes

                nodes = jax.lax.cond(
                    child.log_probability >= self.log_prob_threshold, add_child, do_nothing, (nodes, child)
                )
                return nodes, node

            return jax.lax.fori_loop(0, self.ghmm.num_observations, maybe_add_child, nodes_node)

        def no_update(
            nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
        ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
            return nodes_node

        nodes, node = nodes.remove()
        return jax.lax.cond(node.sequence_length < node.max_sequence_length, add_children, no_update, (nodes, node))

    def get_all_children(self, search_nodes: Queue[MixedStateNode]) -> Queue[MixedStateNode]:
        """Return a queue that contains just contains all the children of the current nodes in the queue."""

        def add_children(_: int, nodes: Queue[MixedStateNode]) -> Queue[MixedStateNode]:
            nodes, _ = self._next_node(nodes)  # type: ignore
            return cast(Queue[MixedStateNode], nodes)

        initial_size = search_nodes.size
        search_nodes = jax.lax.fori_loop(0, initial_size, add_children, search_nodes)
        return search_nodes
