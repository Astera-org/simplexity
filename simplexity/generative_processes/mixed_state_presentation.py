import functools
from enum import Enum
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.data_structures import Collection, Queue, Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.log_math import entropy

Sequence = tuple[int, ...]


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


def compute_average_entropy(dists: jax.Array, probs: jax.Array, size: jax.Array, log: bool = False) -> jax.Array:
    """Compute the weighted average entropy of a collection of distributions."""
    entropies = eqx.filter_vmap(functools.partial(entropy, log=log))(dists)
    if log:
        probs = jnp.exp(probs)
    mask = jnp.where(jnp.arange(dists.shape[0]) < size, 1, 0)
    return jnp.sum(entropies * probs * mask)


def log_compute_average_entropy(log_dists: jax.Array, log_probs: jax.Array, size: jax.Array) -> jax.Array:
    """Compute the weighted average entropy of a collection of distributions."""
    entropies = eqx.filter_vmap(functools.partial(entropy, log=True))(log_dists)
    mask = jnp.where(jnp.arange(log_dists.shape[0]) < size, 1, 0)
    return jnp.sum(entropies * jnp.exp(log_probs) * mask)


class MixedStateNode(eqx.Module):
    """A node in a mixed state presentation of a generative process."""

    sequence: jax.Array
    sequence_length: jax.Array
    unnormalized_belief_state: jax.Array
    belief_state: jax.Array
    probability: jax.Array

    @property
    def num_states(self) -> int:
        """The number of states in the node."""
        return self.belief_state.shape[0]

    @property
    def max_sequence_length(self) -> int:
        """The maximum length of the sequence."""
        return self.sequence.shape[0]


class TreeData(eqx.Module):
    """Data for a tree."""

    sequences: jax.Array
    sequence_lengths: jax.Array
    belief_states: jax.Array
    probabilities: jax.Array
    size: jax.Array

    @classmethod
    def empty(cls, max_size: int, max_sequence_length: int, num_states: int) -> "TreeData":
        """Create an empty tree."""
        return cls(
            sequences=jnp.zeros((max_size, max_sequence_length), dtype=jnp.int32),
            sequence_lengths=jnp.zeros((max_size,), dtype=jnp.int32),
            belief_states=jnp.zeros((max_size, num_states), dtype=jnp.float32),
            probabilities=jnp.zeros((max_size,), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def add(self, node: MixedStateNode) -> "TreeData":
        """Add a sequence to the tree."""
        return TreeData(
            sequences=self.sequences.at[self.size].set(node.sequence),
            sequence_lengths=self.sequence_lengths.at[self.size].set(node.sequence_length),
            belief_states=self.belief_states.at[self.size].set(node.belief_state),
            probabilities=self.probabilities.at[self.size].set(node.probability),
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
        return self.belief_states.shape[1]

    def save(self, path: Path) -> None:
        """Save the tree data to a file."""
        jnp.savez(
            path,
            sequences=self.sequences,
            sequence_lengths=self.sequence_lengths,
            belief_states=self.belief_states,
            probabilities=self.probabilities,
            size=self.size,
        )

    @classmethod
    def load(cls, path: Path) -> "TreeData":
        """Load the tree data from a file."""
        data = jnp.load(path)
        return cls(
            sequences=data["sequences"],
            sequence_lengths=data["sequence_lengths"],
            belief_states=data["belief_states"],
            probabilities=data["probabilities"],
            size=data["size"],
        )


class BaseNodeDictValue(NamedTuple):
    """The value of a node in the node dictionary."""


class NodeDictValue(NamedTuple):
    """The value of a node in the node dictionary."""

    probability: float
    belief_state: tuple[float, ...]


TTreeData = TypeVar("TTreeData", bound=TreeData)
TNodeDictValue = TypeVar("TNodeDictValue", bound=tuple[float, tuple[float, ...]])


class MixedStateTree(Generic[TTreeData, TNodeDictValue]):
    """A presentation of a generative process as a mixed state."""

    def __init__(self, nodes: TTreeData):
        self.nodes: dict[Sequence, TNodeDictValue] = {}
        for i in range(nodes.size):
            self.add(
                nodes.sequences[i, : nodes.sequence_lengths[i]],
                nodes.probabilities[i],
                nodes.belief_states[i],
            )

    def __len__(self) -> int:
        """The number of nodes in the tree."""
        return len(self.nodes)

    def add(self, sequence: jax.Array, probability: jax.Array, belief_state: jax.Array) -> None:
        """Add a sequence to the tree."""
        sequence_ = tuple(sequence.tolist())
        probability_ = probability.item()
        belief_state_ = tuple(belief_state.tolist())
        self.nodes[sequence_] = cast(TNodeDictValue, NodeDictValue(probability_, belief_state_))


# Define type variables for the node and tree types
TNode = TypeVar("TNode", bound=MixedStateNode)
TTreeData = TypeVar("TTreeData", bound=TreeData)
TTree = TypeVar("TTree", bound=MixedStateTree)


class MixedStateTreeGenerator(eqx.Module, Generic[TNode, TTreeData, TTree]):
    """A generator of nodes in a mixed state presentation of a generative process."""

    ghmm: GeneralizedHiddenMarkovModel
    max_sequence_length: int
    max_tree_size: int
    max_search_nodes_size: int
    prob_threshold: float

    def __init__(
        self,
        ghmm: GeneralizedHiddenMarkovModel,
        max_sequence_length: int,
        max_tree_size: int = -1,
        max_search_nodes_size: int = -1,
        prob_threshold: float = 0.0,
    ):
        self.ghmm = ghmm
        self.max_sequence_length = max_sequence_length
        self.max_tree_size = max_tree_size
        self.max_search_nodes_size = max_search_nodes_size
        self.prob_threshold = prob_threshold

    def generate(self, search_algorithm: SearchAlgorithm = SearchAlgorithm.DEPTH_FIRST) -> TTree:
        """Generate all nodes in the tree."""
        tree_data = self._generate_tree_data(search_algorithm)
        return cast(TTree, MixedStateTree(tree_data))

    def _generate_tree_data(self, search_algorithm: SearchAlgorithm) -> TTreeData:
        def continue_loop(carry: tuple[TTreeData, Collection[TNode]]) -> jax.Array:
            tree_data, search_nodes = carry
            return jnp.logical_and(~search_nodes.is_empty, tree_data.size < tree_data.max_size)

        def add_next_node(
            carry: tuple[TTreeData, Collection[TNode]],
        ) -> tuple[TTreeData, Collection[TNode]]:
            tree_data, search_nodes = carry
            search_nodes, node = self._next_node(search_nodes)
            tree_data = tree_data.add(node)
            return tree_data, search_nodes

        if self.max_tree_size < 0:
            max_tree_size = int(jnp.sum(self.ghmm.num_observations ** jnp.arange(self.max_sequence_length + 1)))
        else:
            max_tree_size = self.max_tree_size
        tree_data = self._empty_tree_data(max_tree_size)

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
        return tree_data

    def _compute_entropies(self, search_nodes: Queue[TNode]) -> tuple[jax.Array, jax.Array]:
        """Compute the entropies of the generative process."""
        obs_dist_fn = eqx.filter_vmap(self.ghmm.observation_probability_distribution)
        data = cast(TNode, search_nodes.data)
        obs_dists = obs_dist_fn(data.belief_state)
        belief_state_entropy = compute_average_entropy(data.belief_state, data.probability, search_nodes.size)
        observation_entropy = compute_average_entropy(obs_dists, data.probability, search_nodes.size)
        return belief_state_entropy, observation_entropy

    def compute_myopic_entropy(self) -> MyopicEntropies:
        """Compute the myopic entropy of the generative process."""

        def update_myopic_entropies(
            sequence_length: int, carry: tuple[jax.Array, jax.Array, Queue[TNode]]
        ) -> tuple[jax.Array, jax.Array, Queue[TNode]]:
            belief_state_entropies, observation_entropies, search_nodes = carry
            belief_state_entropy, observation_entropy = self._compute_entropies(search_nodes)
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

    def _empty_tree_data(self, max_size: int) -> TTreeData:
        """Create an empty tree data."""
        return cast(TTreeData, TreeData.empty(max_size, self.max_sequence_length, self.ghmm.num_states))

    @property
    def root(self) -> TNode:
        """The root node of the tree."""
        empty_sequence = jnp.zeros((self.max_sequence_length,), dtype=jnp.int32)
        sequence_length = jnp.array(0)
        unnormalized_belief_state = self.ghmm.stationary_state
        belief_state = self.ghmm.normalize_belief_state(unnormalized_belief_state)
        probability = unnormalized_belief_state @ self.ghmm.normalizing_eigenvector
        return cast(
            TNode, MixedStateNode(empty_sequence, sequence_length, unnormalized_belief_state, belief_state, probability)
        )

    @eqx.filter_jit
    def get_child(self, node: TNode, obs: jax.Array) -> TNode:
        """Get the child of a node."""
        sequence = node.sequence.at[node.sequence_length].set(obs)
        sequence_length = node.sequence_length + 1
        unnormalized_belief_state = node.unnormalized_belief_state @ self.ghmm.transition_matrices[obs]
        belief_state = self.ghmm.normalize_belief_state(unnormalized_belief_state)
        probability = unnormalized_belief_state @ self.ghmm.normalizing_eigenvector
        return cast(
            TNode, MixedStateNode(sequence, sequence_length, unnormalized_belief_state, belief_state, probability)
        )

    @eqx.filter_jit
    def _next_node(self, nodes: Collection[TNode]) -> tuple[Collection[TNode], TNode]:
        """Get the next node from a collection and add that node's children to the collection."""

        def add_children(
            nodes_node: tuple[Collection[TNode], TNode],
        ) -> tuple[Collection[TNode], TNode]:
            def maybe_add_child(i: int, nodes_node: tuple[Collection[TNode], TNode]) -> tuple[Collection[TNode], TNode]:
                nodes, node = nodes_node
                obs = jnp.array(i)
                child = self.get_child(node, obs)

                def add_child(
                    nodes_node: tuple[Collection[TNode], TNode],
                ) -> Collection[TNode]:
                    nodes, node = nodes_node
                    return nodes.add(node)

                def do_nothing(
                    nodes_node: tuple[Collection[TNode], TNode],
                ) -> Collection[TNode]:
                    nodes, _ = nodes_node
                    return nodes

                nodes = jax.lax.cond(child.probability >= self.prob_threshold, add_child, do_nothing, (nodes, child))
                return nodes, node

            return jax.lax.fori_loop(0, self.ghmm.num_observations, maybe_add_child, nodes_node)

        def no_update(
            nodes_node: tuple[Collection[TNode], TNode],
        ) -> tuple[Collection[TNode], TNode]:
            return nodes_node

        nodes, node = nodes.remove()
        return jax.lax.cond(node.sequence_length < node.max_sequence_length, add_children, no_update, (nodes, node))

    def get_all_children(self, search_nodes: Queue[TNode]) -> Queue[TNode]:
        """Return a queue that contains just contains all the children of the current nodes in the queue."""

        def add_children(_: int, nodes: Queue[TNode]) -> Queue[TNode]:
            nodes, _ = self._next_node(nodes)  # type: ignore
            return cast(Queue[TNode], nodes)

        initial_size = search_nodes.size
        search_nodes = jax.lax.fori_loop(0, initial_size, add_children, search_nodes)
        return search_nodes


class LogMixedStateNode(MixedStateNode):
    """A node in a mixed state presentation of a generative process."""

    log_unnormalized_belief_state: jax.Array
    log_belief_state: jax.Array
    log_probability: jax.Array


class LogTreeData(TreeData):
    """Data for a tree."""

    log_belief_states: jax.Array
    log_probabilities: jax.Array

    @classmethod
    def empty(cls, max_size: int, max_sequence_length: int, num_states: int) -> "TreeData":
        """Create an empty tree."""
        return cls(
            sequences=jnp.zeros((max_size, max_sequence_length), dtype=jnp.int32),
            sequence_lengths=jnp.zeros((max_size,), dtype=jnp.int32),
            belief_states=jnp.zeros((max_size, num_states), dtype=jnp.float32),
            probabilities=jnp.zeros((max_size,), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32),
            log_belief_states=jnp.zeros((max_size, num_states), dtype=jnp.float32),
            log_probabilities=jnp.zeros((max_size,), dtype=jnp.float32),
        )

    def add(self, node: MixedStateNode) -> "LogTreeData":
        """Add a sequence to the tree."""
        if isinstance(node, LogMixedStateNode):
            return LogTreeData(
                sequences=self.sequences.at[self.size].set(node.sequence),
                sequence_lengths=self.sequence_lengths.at[self.size].set(node.sequence_length),
                belief_states=self.belief_states.at[self.size].set(node.belief_state),
                probabilities=self.probabilities.at[self.size].set(node.probability),
                size=self.size + 1,
                log_belief_states=self.log_belief_states.at[self.size].set(node.log_belief_state),
                log_probabilities=self.log_probabilities.at[self.size].set(node.log_probability),
            )
        log_belief_state = jnp.log(node.belief_state)
        log_probability = jnp.log(node.probability)
        return LogTreeData(
            sequences=self.sequences.at[self.size].set(node.sequence),
            sequence_lengths=self.sequence_lengths.at[self.size].set(node.sequence_length),
            belief_states=self.belief_states.at[self.size].set(node.belief_state),
            probabilities=self.probabilities.at[self.size].set(node.probability),
            size=self.size + 1,
            log_belief_states=self.log_belief_states.at[self.size].set(log_belief_state),
            log_probabilities=self.log_probabilities.at[self.size].set(log_probability),
        )

    def save(self, path: Path) -> None:
        """Save the tree data to a file."""
        jnp.savez(
            path,
            sequences=self.sequences,
            sequence_lengths=self.sequence_lengths,
            belief_states=self.belief_states,
            probabilities=self.probabilities,
            size=self.size,
            log_belief_states=self.log_belief_states,
            log_probabilities=self.log_probabilities,
        )

    @classmethod
    def load(cls, path: Path) -> "LogTreeData":
        """Load the tree data from a file."""
        data = jnp.load(path)
        return cls(
            sequences=data["sequences"],
            sequence_lengths=data["sequence_lengths"],
            belief_states=data["belief_states"],
            probabilities=data["probabilities"],
            size=data["size"],
            log_belief_states=data["log_belief_states"],
            log_probabilities=data["log_probabilities"],
        )


class LogNodeDictValue(NamedTuple):
    """The value of a node in the node dictionary."""

    log_probability: float
    log_belief_state: tuple[float, ...]


class LogMixedStateTree(MixedStateTree[LogTreeData, LogNodeDictValue]):
    """A presentation of a generative process as a mixed state."""

    def __init__(self, nodes: LogTreeData):
        self.nodes: dict[Sequence, LogNodeDictValue] = {}
        for i in range(nodes.size):
            self.add(
                nodes.sequences[i, : nodes.sequence_lengths[i]],
                nodes.log_probabilities[i],
                nodes.log_belief_states[i],
            )

    def __len__(self) -> int:
        """The number of nodes in the tree."""
        return len(self.nodes)

    def add(self, sequence: jax.Array, probability: jax.Array, belief_state: jax.Array) -> None:
        """Add a sequence to the tree."""
        sequence_ = tuple(sequence.tolist())
        log_probability_ = float(probability.item())
        log_belief_state_ = tuple(belief_state.tolist())
        self.nodes[sequence_] = LogNodeDictValue(log_probability_, log_belief_state_)


class LogMixedStateTreeGenerator(MixedStateTreeGenerator[LogMixedStateNode, LogTreeData, LogMixedStateTree]):
    """A generator of nodes in a mixed state presentation of a generative process."""

    # TODO: enable GHMM support, https://github.com/Astera-org/simplexity/issues/9
    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        assert isinstance(self.ghmm, HiddenMarkovModel), "LogMixedStateTreeGenerator only works with HiddenMarkovModels"

    def generate(self, search_algorithm: SearchAlgorithm = SearchAlgorithm.DEPTH_FIRST) -> LogMixedStateTree:
        """Generate all nodes in the tree."""
        tree_data = self._generate_tree_data(search_algorithm)
        return cast(LogMixedStateTree, LogMixedStateTree(tree_data))

    def _compute_entropies(self, search_nodes: Queue[LogMixedStateNode]) -> tuple[jax.Array, jax.Array]:
        """Compute the entropies of the generative process."""
        log_obs_dist_fn = eqx.filter_vmap(self.ghmm.log_observation_probability_distribution)
        data = cast(LogMixedStateNode, search_nodes.data)
        log_obs_dists = log_obs_dist_fn(data.log_belief_state)
        belief_state_entropy = compute_average_entropy(
            data.log_belief_state, data.log_probability, search_nodes.size, log=True
        )
        observation_entropy = compute_average_entropy(log_obs_dists, data.log_probability, search_nodes.size, log=True)
        return belief_state_entropy, observation_entropy

    def _empty_tree_data(self, max_size: int) -> LogTreeData:
        """Create an empty tree data."""
        return cast(LogTreeData, LogTreeData.empty(max_size, self.max_sequence_length, self.ghmm.num_states))

    @property
    def root(self) -> LogMixedStateNode:
        """The root node of the tree."""
        empty_sequence = jnp.zeros((self.max_sequence_length,), dtype=jnp.int32)
        sequence_length = jnp.array(0)
        unnormalized_belief_state = self.ghmm.stationary_state
        belief_state = self.ghmm.normalize_belief_state(unnormalized_belief_state)
        probability = unnormalized_belief_state @ self.ghmm.normalizing_eigenvector
        log_unnormalized_belief_state = self.ghmm.log_stationary_state
        log_belief_state = self.ghmm.normalize_log_belief_state(log_unnormalized_belief_state)
        log_probability = jax.nn.logsumexp(log_unnormalized_belief_state + self.ghmm.log_normalizing_eigenvector)
        return LogMixedStateNode(
            empty_sequence,
            sequence_length,
            unnormalized_belief_state,
            belief_state,
            probability,
            log_unnormalized_belief_state,
            log_belief_state,
            log_probability,
        )

    @eqx.filter_jit
    def get_child(self, node: LogMixedStateNode, obs: jax.Array) -> LogMixedStateNode:
        """Get the child of a node."""
        parent_log_unnormalized_belief_state = node.log_unnormalized_belief_state
        sequence = node.sequence.at[node.sequence_length].set(obs)
        sequence_length = node.sequence_length + 1
        log_unnormalized_belief_state = jax.nn.logsumexp(
            parent_log_unnormalized_belief_state[:, None] + self.ghmm.log_transition_matrices[obs], axis=0
        )
        log_belief_state = self.ghmm.normalize_log_belief_state(log_unnormalized_belief_state)
        log_probability = jax.nn.logsumexp(log_unnormalized_belief_state + self.ghmm.log_normalizing_eigenvector)
        unnormalized_belief_state = jnp.exp(log_unnormalized_belief_state)
        belief_state = jnp.exp(log_belief_state)
        probability = jnp.exp(log_probability)
        return LogMixedStateNode(
            sequence,
            sequence_length,
            unnormalized_belief_state,
            belief_state,
            probability,
            log_unnormalized_belief_state,
            log_belief_state,
            log_probability,
        )
