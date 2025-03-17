import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator, SearchAlgorithm
from simplexity.generative_processes.transition_matrices import (
    fanizza,
    mess3,
    post_quantum,
    rrxor,
    tom_quantum,
    zero_one_random,
)

golden_file_path = Path(__file__).parent / "goldens" / "msps"

# Fixed tolerance for all comparisons
TOLERANCE = 1e-5

# Dictionary of HMM golden files
HMM_GOLDENS = {
    "mess3": golden_file_path / "mess3_x_0.15_a_0.6.npz",
    "rrxor": golden_file_path / "rrxor_pR1_0.5_pR2_0.5.npz",
    "zero_one_random": golden_file_path / "zero_one_random_p_0.5.npz",
}

# Dictionary of GHMM golden files
GHMM_GOLDENS = {
    # Specialized GHMM processes
    "fanizza": golden_file_path / "fanizza_alpha_2000_lamb_0.49.npz",
    "post_quantum": golden_file_path / "post_quantum_alpha_2.7_beta_0.5.npz",
    # HMM processes represented as GHMM
    "ghmm_mess3": golden_file_path / "mess3_x_0.15_a_0.6.npz",
    "ghmm_rrxor": golden_file_path / "rrxor_pR1_0.5_pR2_0.5.npz",
    "ghmm_zero_one_random": golden_file_path / "zero_one_random_p_0.5.npz",
}

# Filter out any files that don't exist
HMM_GOLDENS = {k: v for k, v in HMM_GOLDENS.items() if v.exists()}
GHMM_GOLDENS = {k: v for k, v in GHMM_GOLDENS.items() if v.exists()}


def reconstruct_node_info(npz_file: str | Path) -> dict[tuple[int, ...], dict[str, float | jnp.ndarray]]:
    """Load a golden MSP file and convert it to a dictionary of node information."""
    # Convert to string if Path object
    npz_file_str = str(npz_file)
    data = np.load(npz_file_str)
    paths = data["paths"]
    probs = data["probs"]
    beliefs = data["beliefs"]

    node_info = {}
    for i in range(len(paths)):
        # Convert path array to tuple, ignoring padding (-1 values)
        path = tuple(int(x) for x in paths[i] if x >= 0)

        node_info[path] = {"path_prob": float(probs[i]), "belief_state": beliefs[i]}

    return node_info


def get_generator_for_process(process_key: str) -> tuple[MixedStateTreeGenerator, str, dict[str, float]]:
    """Create a MixedStateTreeGenerator for a specific process.

    Args:
        process_key: A process name from HMM_GOLDENS or GHMM_GOLDENS dictionaries.

    Returns:
        A tuple of (generator, process_name, parameters_dict)
    """
    # Check if it's a valid process key
    is_hmm = process_key in HMM_GOLDENS
    is_ghmm = process_key in GHMM_GOLDENS

    if not (is_hmm or is_ghmm):
        raise ValueError(
            f"Unknown process key: {process_key}. Must be one of: {list(HMM_GOLDENS.keys()) + list(GHMM_GOLDENS.keys())}"
        )

    # Set max_sequence_length to 4 (as mentioned in the README)
    max_sequence_length = 4

    # Define parameter mappings for all processes
    PROCESS_PARAMS = {
        "mess3": {"x": 0.15, "a": 0.6},
        "zero_one_random": {"p": 0.5},
        "rrxor": {"pR1": 0.5, "pR2": 0.5},
        "fanizza": {"alpha": 2000, "lamb": 0.49},
        "post_quantum": {"log_alpha": np.log(2.7), "beta": 0.5},
        "tom_quantum": {"alpha": 1, "beta": 0.5},
    }

    # Check if this is a GHMM version of an HMM process
    base_process = process_key

    if process_key.startswith("ghmm_"):
        base_process = process_key[5:]  # Remove 'ghmm_' prefix

    # Get parameters for the base process
    if base_process not in PROCESS_PARAMS:
        raise ValueError(f"Process {base_process} not supported in this test")

    params = PROCESS_PARAMS[base_process]

    # Create transition matrices based on the base process
    if base_process == "mess3":
        transition_matrices = mess3(**params)
    elif base_process == "zero_one_random":
        transition_matrices = zero_one_random(**params)
    elif base_process == "rrxor":
        transition_matrices = rrxor(**params)
    elif base_process == "fanizza":
        transition_matrices = fanizza(**params)
    elif base_process == "post_quantum":
        transition_matrices = post_quantum(**params)
    elif base_process == "tom_quantum":
        transition_matrices = tom_quantum(**params)
    else:
        raise ValueError(f"Process {base_process} not supported in this test")

    # Create the appropriate model and generator
    if is_ghmm:
        model = GeneralizedHiddenMarkovModel(transition_matrices)
    else:
        model = HiddenMarkovModel(transition_matrices)

    generator = MixedStateTreeGenerator(model, max_sequence_length=max_sequence_length)

    # Return the generator, process name, and parameters
    return generator, process_key, params


def test_against_golden_with_generator(golden_file, generator):
    """Run the golden test with a pre-configured generator.

    Args:
        golden_file: Path to the golden file to test against.
        generator: The pre-configured MixedStateTreeGenerator.
        rel_tolerance: Relative tolerance for comparing belief states (only used with relaxed=True).
    """
    # Load golden data
    golden_node_info = reconstruct_node_info(golden_file)

    # Generate tree
    tree = generator.generate(search_algorithm=SearchAlgorithm.BREADTH_FIRST)

    # Check that all sequences from golden data are in the generated tree
    golden_sequences = set(golden_node_info.keys())

    # Convert tree.nodes to use the same key type
    generated_sequences = {tuple(seq) if isinstance(seq, tuple) else seq for seq in tree.nodes.keys()}

    # Check if the sequences match
    missing_sequences = golden_sequences - generated_sequences
    extra_sequences = generated_sequences - golden_sequences

    assert not missing_sequences, (
        f"Generated tree is missing sequences from golden data for {golden_file.name}: {missing_sequences}"
    )

    # Special case for zero_one_random which might have extra sequences
    is_zero_one_random = str(golden_file).find("zero_one_random") != -1

    if is_zero_one_random and extra_sequences:
        # For zero_one_random, check that any extra sequences have zero or near-zero probability
        zero_prob_threshold = 1e-10
        for sequence in extra_sequences:
            # Make sure we're using the right type of sequence for the lookup
            tree_seq = sequence  # Use as-is for lookup in tree.nodes
            generated_log_prob = tree.nodes[tree_seq].log_probability
            generated_prob = math.exp(generated_log_prob) if generated_log_prob != -math.inf else 0
            assert generated_prob < zero_prob_threshold, (
                f"Extra sequence {sequence} in {golden_file.name} has non-zero probability: {generated_prob}"
            )
    else:
        # For other processes, there should be no extra sequences
        assert not extra_sequences, (
            f"Generated tree has extra sequences not in golden data for {golden_file.name}: {extra_sequences}"
        )

    # Compare probabilities and belief states for each sequence
    for sequence in golden_node_info:
        # Get the golden values
        golden_prob = golden_node_info[sequence]["path_prob"]
        golden_belief = golden_node_info[sequence]["belief_state"]

        # Get the generated values (converting from log if needed)
        tree_seq = sequence  # Use as-is for lookup in tree.nodes
        generated_log_prob = tree.nodes[tree_seq].log_probability
        generated_prob = math.exp(generated_log_prob) if generated_log_prob != -math.inf else 0

        # Compare probabilities
        assert math.isclose(generated_prob, golden_prob, abs_tol=TOLERANCE), (
            f"Probability mismatch for sequence {sequence} in {golden_file.name}: "
            f"got {generated_prob}, expected {golden_prob}"
        )

        # Always compare belief states with the fixed tolerance
        generated_log_belief = tree.nodes[tree_seq].log_belief_state
        # Convert log belief to regular belief, handling -inf -> 0
        generated_belief = []
        for log_val in generated_log_belief:
            if math.isnan(log_val):
                generated_belief.append(math.nan)
            elif log_val == -math.inf:
                generated_belief.append(0.0)
            else:
                generated_belief.append(math.exp(log_val))

        # Compare belief states
        compare_belief_states(generated_belief, golden_belief, sequence, golden_file.name)


def compare_belief_states(generated_belief, golden_belief, sequence, file_name):
    """Compare belief states, handling different formats.

    The golden belief states might be either:
    1. A 1D array
    2. A 2D array with shape (1, n)

    We need to handle both cases.

    Args:
        generated_belief: The belief state from the generated tree.
        golden_belief: The belief state from the golden file.
        sequence: The sequence being compared.
        file_name: The name of the golden file.
    """
    # Convert golden belief to 1D if it's 2D
    if len(golden_belief.shape) > 1:
        golden_belief = golden_belief.flatten()

    # First check if sizes match
    assert len(generated_belief) == len(golden_belief), (
        f"Belief state length mismatch for sequence {sequence} in {file_name}: "
        f"got {len(generated_belief)}, expected {len(golden_belief)}"
    )

    # Then check values
    for i, (gen_val, gold_val) in enumerate(zip(generated_belief, golden_belief, strict=False)):
        if math.isnan(gold_val):
            assert math.isnan(gen_val), (
                f"Belief state value mismatch for sequence {sequence} at index {i} in {file_name}: "
                f"got {gen_val}, expected NaN"
            )
        else:
            try:
                # Always use fixed tolerance
                assert math.isclose(gen_val, gold_val, abs_tol=TOLERANCE), (
                    f"Belief state value mismatch for sequence {sequence} at index {i} in {file_name}: "
                    f"got {gen_val}, expected {gold_val}"
                )
            except AssertionError:
                print(f"\nMismatch for sequence {sequence} at index {i}")
                print(f"Generated value: {gen_val}")
                print(f"Golden value: {gold_val}")
                diff = abs(gen_val - gold_val)
                print(f"Absolute difference: {diff}")
                raise


# List of all HMM process keys to test
@pytest.fixture(params=HMM_GOLDENS.keys())
def hmm_process_key(request):
    return request.param


# List of all GHMM process keys to test
@pytest.fixture(params=GHMM_GOLDENS.keys())
def ghmm_process_key(request):
    return request.param


@pytest.mark.parametrize("process_key", list(HMM_GOLDENS.keys()))
def test_hmm_golden(process_key):
    """Test that each HMM process matches its golden data."""
    golden_file = HMM_GOLDENS[process_key]
    generator, _, _ = get_generator_for_process(process_key)
    test_against_golden_with_generator(golden_file, generator)


@pytest.mark.parametrize("process_key", list(GHMM_GOLDENS.keys()))
def test_ghmm_golden(process_key):
    """Test that each GHMM process matches its golden data."""
    golden_file = GHMM_GOLDENS[process_key]
    generator, _, _ = get_generator_for_process(process_key)
    test_against_golden_with_generator(golden_file, generator)
