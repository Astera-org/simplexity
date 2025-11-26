"""Test the mixed-state tree module."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_generalized_hidden_markov_model, build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import (
    MixedStateTree,
    MixedStateTreeGenerator,
    NodeDictValue,
    TreeData,
)

GOLDEN_DIR = Path(__file__).parent / "goldens" / "mixed_state_trees"

PROCESS_PARAMS: dict[str, dict[str, float | int]] = {
    "fanizza": {
        "alpha": 2000,
        "lamb": 0.49,
    },
    "mess3": {
        "x": 0.15,
        "a": 0.6,
    },
    "post_quantum": {
        "log_alpha": float(jnp.log(2.7)),
        "beta": 0.5,
    },
    "rrxor": {
        "p1": 0.5,
        "p2": 0.5,
    },
    "tom_quantum": {
        "alpha": 1,
        "beta": 4.3,
    },
    "zero_one_random": {
        "p": 0.5,
    },
}

GHMM_PROCESSES = list(PROCESS_PARAMS.keys())


def filename_value(value: float | int, max_precision: int = 10) -> str:
    """Convert a float or int to a string that is safe for use in a filename."""
    if isinstance(value, int):
        return str(value)
    s = f"{value:.{max_precision}f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def golden_file_name(process_name: str) -> str:
    """Generate a filename for a golden file for a given process."""
    if process_name == "post_quantum":
        return "post_quantum_alpha_2p7_beta_0p5.npz"
    params = PROCESS_PARAMS[process_name]
    if not params:
        return f"{process_name}.npz"
    params_str = "_".join(f"{k}_{filename_value(v)}" for k, v in params.items())
    return f"{process_name}_{params_str}.npz"


def load_golden(process_name: str) -> MixedStateTree[TreeData, NodeDictValue]:
    """Load a golden file for a given process."""
    file_name = golden_file_name(process_name)
    golden_file = GOLDEN_DIR / file_name
    tree_data = TreeData.load(golden_file)
    return MixedStateTree(tree_data)


@pytest.mark.parametrize("process_name", ["mess3", "rrxor", "zero_one_random"])
def test_hmm_mixed_state_tree(process_name: str):
    """Test generating a mixed-state tree for a hidden Markov model."""
    params = PROCESS_PARAMS[process_name]
    model = build_hidden_markov_model(process_name, process_params=params, initial_state=None)
    generator = MixedStateTreeGenerator(model, max_sequence_length=4)
    tree: MixedStateTree[TreeData, NodeDictValue] = generator.generate()
    golden = load_golden(process_name)

    sequences = set(sequence for sequence, values in tree.nodes.items() if values.probability > 0)
    expected_sequences = set(sequence for sequence, values in golden.nodes.items() if values.probability > 0)
    assert sequences == expected_sequences

    for sequence in sequences:
        probability = tree.nodes[sequence].probability
        expected_probability = golden.nodes[sequence].probability
        assert jnp.isclose(probability, expected_probability)

        belief_state = tree.nodes[sequence].belief_state
        expected_belief_state = golden.nodes[sequence].belief_state
        chex.assert_trees_all_close(belief_state, expected_belief_state)


@pytest.mark.parametrize(
    "process_name", ["fanizza", "mess3", "post_quantum", "rrxor", "tom_quantum", "zero_one_random"]
)
def test_ghmm_mixed_state_tree(process_name: str):
    """Test generating a mixed-state tree for a generalized hidden Markov model."""
    params = PROCESS_PARAMS[process_name]
    model = build_generalized_hidden_markov_model(process_name, process_params=params)
    generator = MixedStateTreeGenerator(model, max_sequence_length=4)
    tree: MixedStateTree[TreeData, NodeDictValue] = generator.generate()
    golden = load_golden(process_name)

    sequences = set(sequence for sequence, values in tree.nodes.items() if values.probability > 1e-6)
    expected_sequences = set(sequence for sequence, values in golden.nodes.items() if values.probability > 1e-6)
    assert sequences == expected_sequences

    for sequence in sequences:
        probability = tree.nodes[sequence].probability
        expected_probability = golden.nodes[sequence].probability
        assert jnp.isclose(probability, expected_probability, atol=1e-6)

        belief_state = tree.nodes[sequence].belief_state
        expected_belief_state = golden.nodes[sequence].belief_state
        chex.assert_trees_all_close(belief_state, expected_belief_state, atol=1e-6)
