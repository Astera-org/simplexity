"""Builder for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import inspect
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp

from simplexity.generative_processes.factored_generative_process import ComponentType, FactoredGenerativeProcess
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.independent_factored_generative_process import IndependentFactoredGenerativeProcess
from simplexity.generative_processes.inflated_vocabulary_process import InflatedVocabularyProcess
from simplexity.generative_processes.nonergodic_generative_process import NonErgodicGenerativeProcess
from simplexity.generative_processes.structures import (
    ConditionalTransitions,
    FullyConditional,
    IndependentStructure,
    SequentialConditional,
)
from simplexity.generative_processes.transition_matrices import (
    GHMM_MATRIX_FUNCTIONS,
    HMM_MATRIX_FUNCTIONS,
    get_stationary_state,
)
from simplexity.utils.jnp_utils import resolve_jax_device


def build_transition_matrices(
    matrix_functions: dict[str, Callable],
    process_name: str,
    process_params: Mapping[str, Any] | None = None,
    device: str | None = None,
) -> jax.Array:
    """Build transition matrices for a generative process."""
    if process_name not in matrix_functions:
        raise KeyError(
            f'Unknown process type: "{process_name}".  '
            f"Available HMM processes are: {', '.join(matrix_functions.keys())}"
        )
    matrix_function = matrix_functions[process_name]
    process_params = process_params or {}
    sig = inspect.signature(matrix_function)
    jax_device = resolve_jax_device(device)
    try:
        with jax.default_device(jax_device):
            sig.bind_partial(**process_params)
            transition_matrices = matrix_function(**process_params)
    except TypeError as e:
        params = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(f"Invalid arguments for {process_name}: {e}.  Signature is: {params}") from e

    return transition_matrices


def add_begin_of_sequence_token(transition_matrix: jax.Array, initial_state: jax.Array | None = None) -> jax.Array:
    """Augments transition matrices with a BOS token."""
    base_vocab_size, num_states, _ = transition_matrix.shape
    augmented_matrix = jnp.zeros((base_vocab_size + 1, num_states + 1, num_states + 1), dtype=transition_matrix.dtype)
    augmented_matrix = augmented_matrix.at[:base_vocab_size, :num_states, :num_states].set(transition_matrix)
    if initial_state is None:
        initial_state = get_stationary_state(transition_matrix.sum(axis=0).T)
    return augmented_matrix.at[base_vocab_size, num_states, :num_states].set(initial_state)


def build_hidden_markov_model(
    process_name: str,
    process_params: Mapping[str, Any] | None = None,
    initial_state: jax.Array | Sequence[float] | None = None,
    device: str | None = None,
    noise_epsilon: float = 0.0,
) -> HiddenMarkovModel:
    """Build a hidden Markov model."""
    process_params = process_params or {}
    initial_state = jnp.array(initial_state) if initial_state is not None else None
    transition_matrices = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, process_params, device=device)
    return HiddenMarkovModel(transition_matrices, initial_state, device=device, noise_epsilon=noise_epsilon)


def build_generalized_hidden_markov_model(
    process_name: str,
    process_params: Mapping[str, Any] | None = None,
    initial_state: jax.Array | Sequence[float] | None = None,
    device: str | None = None,
    noise_epsilon: float = 0.0,
) -> GeneralizedHiddenMarkovModel:
    """Build a generalized hidden Markov model."""
    process_params = process_params or {}
    initial_state = jnp.array(initial_state) if initial_state is not None else None
    transition_matrices = build_transition_matrices(GHMM_MATRIX_FUNCTIONS, process_name, process_params, device=device)
    return GeneralizedHiddenMarkovModel(transition_matrices, initial_state, device=device, noise_epsilon=noise_epsilon)


def build_nonergodic_transition_matrices(
    component_transition_matrices: Sequence[jax.Array], vocab_maps: Sequence[Sequence[int]] | None = None
) -> jax.Array:
    """Build composite transition matrices of a nonergodic process from component transition matrices."""
    if vocab_maps is None:
        vocab_maps = [list(range(matrix.shape[0])) for matrix in component_transition_matrices]
    vocab_size = max(max(vocab_map) for vocab_map in vocab_maps) + 1
    total_states = sum(matrix.shape[1] for matrix in component_transition_matrices)
    composite_transition_matrix = jnp.zeros((vocab_size, total_states, total_states))
    state_offset = 0
    for matrix, vocab_map in zip(component_transition_matrices, vocab_maps, strict=True):
        for component_vocab_idx, composite_vocab_idx in enumerate(vocab_map):
            composite_transition_matrix = composite_transition_matrix.at[
                composite_vocab_idx,
                state_offset : state_offset + matrix.shape[1],
                state_offset : state_offset + matrix.shape[1],
            ].set(matrix[component_vocab_idx])
        state_offset += matrix.shape[1]
    return composite_transition_matrix


def build_nonergodic_initial_state(
    component_initial_states: Sequence[jax.Array], process_weights: jax.Array
) -> jax.Array:
    """Build initial state for a nonergodic process from component initial states."""
    assert process_weights.shape == (len(component_initial_states),)
    assert jnp.all(process_weights >= 0)
    process_probabilities = process_weights / process_weights.sum()
    return jnp.concatenate(
        [p * state for p, state in zip(process_probabilities, component_initial_states, strict=True)], axis=0
    )


def build_nonergodic_hidden_markov_model(
    process_names: list[str],
    process_params: Sequence[Mapping[str, Any]],
    process_weights: Sequence[float],
    vocab_maps: Sequence[Sequence[int]] | None = None,
    add_bos_token: bool = False,
    device: str | None = None,
) -> HiddenMarkovModel:
    """Build a hidden Markov model from a list of process names and their corresponding keyword arguments."""
    component_transition_matrices = [
        build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, process_params, device=device)
        for process_name, process_params in zip(process_names, process_params, strict=True)
    ]
    composite_transition_matrix = build_nonergodic_transition_matrices(component_transition_matrices, vocab_maps)
    component_initial_states = [
        get_stationary_state(transition_matrix.sum(axis=0).T) for transition_matrix in component_transition_matrices
    ]
    initial_state = build_nonergodic_initial_state(component_initial_states, jnp.array(process_weights))
    if add_bos_token:
        composite_transition_matrix = add_begin_of_sequence_token(composite_transition_matrix, initial_state)
        num_states = composite_transition_matrix.shape[1]
        initial_state = jnp.zeros((num_states,), dtype=composite_transition_matrix.dtype)
        initial_state = initial_state.at[num_states - 1].set(1)
    return HiddenMarkovModel(composite_transition_matrix, initial_state, device=device)


def build_factored_process(
    structure_type: Literal["independent", "chain", "symmetric", "transition_coupled"],
    component_types: Sequence[ComponentType],
    transition_matrices: Sequence[jax.Array],
    normalizing_eigenvectors: Sequence[jax.Array],
    initial_states: Sequence[jax.Array],
    noise_epsilon: float = 0.0,
    **structure_kwargs,
) -> FactoredGenerativeProcess:
    """Factory function for building factored processes with different conditional structures.

    Args:
        structure_type: Which conditional structure to instantiate
        component_types: Type of each factor ("hmm" or "ghmm")
        transition_matrices: Per-factor transition tensors (shape [K_i, V_i, S_i, S_i])
        normalizing_eigenvectors: Per-factor eigenvectors (shape [K_i, S_i])
        initial_states: Initial state per factor (shape [S_i])
        noise_epsilon: Noisy channel epsilon value
        **structure_kwargs: Structure-specific keyword arguments:
            - For "independent": (none)
            - For "chain": control_maps
            - For "symmetric": control_maps
            - For "transition_coupled": control_maps_transition,
              emission_variant_indices, emission_control_maps (optional)

    Returns:
        FactoredGenerativeProcess configured with the requested conditional structure

    Raises:
        ValueError: If structure_type is invalid or required kwargs are missing
    """
    vocab_sizes = jnp.array([int(T.shape[1]) for T in transition_matrices])

    if structure_type == "independent":
        structure = IndependentStructure()
        return IndependentFactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            noise_epsilon=noise_epsilon,
        )

    if structure_type == "chain":
        if "control_maps" not in structure_kwargs:
            raise ValueError("Missing required argument 'control_maps' for chain structure")
        structure = SequentialConditional(control_maps=tuple(structure_kwargs["control_maps"]), vocab_sizes=vocab_sizes)
    elif structure_type == "symmetric":
        if "control_maps" not in structure_kwargs:
            raise ValueError("Missing required argument 'control_maps' for symmetric structure")
        structure = FullyConditional(control_maps=tuple(structure_kwargs["control_maps"]), vocab_sizes=vocab_sizes)
    elif structure_type == "transition_coupled":
        if "control_maps_transition" not in structure_kwargs:
            raise ValueError("Missing required argument 'control_maps_transition' for transition_coupled structure")
        if "emission_variant_indices" not in structure_kwargs:
            raise ValueError("Missing required argument 'emission_variant_indices' for transition_coupled structure")
        structure = ConditionalTransitions(
            control_maps_transition=tuple(structure_kwargs["control_maps_transition"]),
            emission_variant_indices=structure_kwargs["emission_variant_indices"],
            vocab_sizes=vocab_sizes,
            emission_control_maps=tuple(structure_kwargs["emission_control_maps"])
            if "emission_control_maps" in structure_kwargs and structure_kwargs["emission_control_maps"] is not None
            else None,
        )
    else:
        raise ValueError(f"Unknown structure_type '{structure_type}'")

    return FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
        noise_epsilon=noise_epsilon,
    )


def build_factored_process_from_spec(
    structure_type: Literal["independent", "chain", "symmetric", "transition_coupled"],
    spec: Sequence[dict[str, Any]],
    noise_epsilon: float = 0.0,
    **structure_params,
) -> FactoredGenerativeProcess:
    """Unified builder for factored processes from specification.

    Args:
        structure_type: Which conditional structure to use
        spec: Component specifications. Format depends on structure_type:
            - For "independent": List of component dicts
            - For "chain": List of component dicts with control_maps
            - For "symmetric": List of component dicts
            - For "transition_coupled": List of component dicts
        noise_epsilon: Noisy channel epsilon value
        **structure_params: Additional structure-specific parameters:
            - For "independent": (none)
            - For "chain": (none, uses spec's control_map fields)
            - For "symmetric": control_maps (list)
            - For "transition_coupled": control_maps_transition, emission_variant_indices,
              emission_control_maps (optional)

    Returns:
        FactoredGenerativeProcess with specified structure

    Example:
        ```python
        # Independent
        process = build_factored_process_from_spec(
            structure_type="independent",
            spec=[
                {"component_type": "hmm", "variants": [{"process_name": "mess3", "x": 0.15, "a": 0.6}]},
                {"component_type": "hmm", "variants": [{"process_name": "mess3", "x": 0.5, "a": 0.6}]},
            ],
        )
        ```
    """
    if structure_type == "independent":
        component_types, transition_matrices, normalizing_eigenvectors, initial_states = build_matrices_from_spec(spec)
        return build_factored_process(
            structure_type="independent",
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            noise_epsilon=noise_epsilon,
        )
    elif structure_type == "chain":
        component_types, transition_matrices, normalizing_eigenvectors, initial_states, control_maps = (
            build_chain_from_spec(spec)
        )
        return build_factored_process(
            structure_type="chain",
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            noise_epsilon=noise_epsilon,
            control_maps=control_maps,
        )
    elif structure_type == "symmetric":
        if "control_maps" not in structure_params:
            raise ValueError("symmetric structure requires 'control_maps' parameter")
        (
            component_types,
            transition_matrices,
            normalizing_eigenvectors,
            initial_states,
            control_maps_arrays,
        ) = build_symmetric_from_spec(spec, structure_params["control_maps"])
        return build_factored_process(
            structure_type="symmetric",
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            noise_epsilon=noise_epsilon,
            control_maps=control_maps_arrays,
        )
    elif structure_type == "transition_coupled":
        if "control_maps_transition" not in structure_params:
            raise ValueError("transition_coupled structure requires 'control_maps_transition' parameter")
        if "emission_variant_indices" not in structure_params:
            raise ValueError("transition_coupled structure requires 'emission_variant_indices' parameter")
        (
            component_types,
            transition_matrices,
            normalizing_eigenvectors,
            initial_states,
            control_maps_arrays,
            emission_variant_indices_array,
            emission_control_maps_arrays,
        ) = build_transition_coupled_from_spec(
            spec,
            structure_params["control_maps_transition"],
            structure_params["emission_variant_indices"],
            structure_params.get("emission_control_maps"),
        )
        return build_factored_process(
            structure_type="transition_coupled",
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            noise_epsilon=noise_epsilon,
            control_maps_transition=control_maps_arrays,
            emission_variant_indices=emission_variant_indices_array,
            emission_control_maps=emission_control_maps_arrays,
        )
    else:
        raise ValueError(f"Unknown structure_type '{structure_type}'")


def build_matrices_from_spec(
    spec: Sequence[dict[str, Any]],
) -> tuple[
    list[ComponentType],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
]:
    """Build transition matrices, eigenvectors, and initial states from spec.

    This is a generic helper that works for all conditional structures. Each element of spec
    should be a dict with:
      - component_type: "hmm" | "ghmm"
      - variants: list of dicts, each with "process_name" and process-specific kwargs

    Args:
        spec: List of factor specifications

    Returns:
        Tuple of (component_types, transition_matrices, normalizing_eigenvectors, initial_states)

    Example:
        ```python
        spec = [
            {
                "component_type": "hmm",
                "variants": [
                    {"process_name": "mess3", "x": 0.15, "a": 0.6},
                    {"process_name": "mess3", "x": 0.5, "a": 0.6},
                ]
            },
            {
                "component_type": "ghmm",
                "variants": [
                    {"process_name": "tom_quantum", "alpha": 1.0, "beta": 1.0},
                ]
            },
        ]
        component_types, T_mats, norms, states = build_matrices_from_spec(spec)
        ```
    """
    if not spec:
        raise ValueError("spec must contain at least one factor")

    component_types: list[ComponentType] = []
    transition_matrices: list[jax.Array] = []
    normalizing_eigenvectors: list[jax.Array] = []
    initial_states: list[jax.Array] = []

    for idx, factor_spec in enumerate(spec):
        ctype: ComponentType = factor_spec.get("component_type", "ghmm")
        variants: Sequence[dict[str, Any]] = factor_spec.get("variants", [])

        if not variants:
            raise ValueError(f"spec[{idx}].variants must be non-empty")

        # Build all variants for this factor
        built = [
            build_hidden_markov_model(**v) if ctype == "hmm" else build_generalized_hidden_markov_model(**v)
            for v in variants
        ]

        # Validate dimensions
        vocab_sizes = [b.vocab_size for b in built]
        num_states = [b.num_states if hasattr(b, "num_states") else b.transition_matrices.shape[1] for b in built]

        if len(set(vocab_sizes)) != 1:
            raise ValueError(f"All variants in spec[{idx}] must have same vocab size; got {vocab_sizes}")
        if len(set(num_states)) != 1:
            raise ValueError(f"All variants in spec[{idx}] must have same state dim; got {num_states}")

        S = num_states[0]

        # Stack transition matrices: [K, V, S, S]
        T_stack = jnp.stack([b.transition_matrices for b in built], axis=0)
        transition_matrices.append(T_stack)

        # Stack normalizing eigenvectors (or create dummy for HMM)
        if ctype == "ghmm":
            norms = jnp.stack([b.normalizing_eigenvector for b in built], axis=0)  # [K, S]
        else:  # dummy (unused) vector for HMM
            norms = jnp.ones((len(built), S))
        normalizing_eigenvectors.append(norms)

        # Initial state: use variant 0's initial state
        initial_states.append(built[0].initial_state)

        component_types.append(ctype)

    return component_types, transition_matrices, normalizing_eigenvectors, initial_states


def build_chain_from_spec(
    chain: Sequence[dict[str, Any]],
) -> tuple[
    list[ComponentType],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array | None],
]:
    """Build all parameters for chain structure from chain specification.

    Each element of chain should be a dict with:
      - component_type: "hmm" | "ghmm"
      - variants: list of variant specs
      - control_map (optional for index 0, required for i>0): list[int] mapping
        parent token -> variant index

    Args:
        chain: List of factor specifications with control maps

    Returns:
        Tuple of (component_types, transition_matrices, normalizing_eigenvectors,
                 initial_states, control_maps)

    Example:
        ```python
        chain = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "x": 0.15, "a": 0.6}],
                # No control_map for root
            },
            {
                "component_type": "hmm",
                "variants": [
                    {"process_name": "mess3", "x": 0.15, "a": 0.6},
                    {"process_name": "mess3", "x": 0.5, "a": 0.6},
                ],
                "control_map": [0, 1, 0],  # Maps 3 parent tokens -> 2 variants
            },
        ]
        ```
    """
    if not chain:
        raise ValueError("chain must contain at least one node")

    # Build base matrices
    component_types, transition_matrices, normalizing_eigenvectors, initial_states = build_matrices_from_spec(chain)

    # Extract control maps
    control_maps: list[jax.Array | None] = []
    expected_prev_vocab = None

    for idx, node in enumerate(chain):
        if idx == 0:
            control_maps.append(None)
        else:
            cm = node.get("control_map", None)
            if cm is None:
                raise ValueError(f"chain[{idx}].control_map is required for i>0")

            cm_arr = jnp.asarray(cm, dtype=jnp.int32)

            if expected_prev_vocab is not None and int(cm_arr.shape[0]) != int(expected_prev_vocab):
                raise ValueError(
                    f"chain[{idx}].control_map length {cm_arr.shape[0]} must equal parent vocab {expected_prev_vocab}"
                )

            control_maps.append(cm_arr)

        # Track vocab size for next iteration
        expected_prev_vocab = int(transition_matrices[idx].shape[1])

    return component_types, transition_matrices, normalizing_eigenvectors, initial_states, control_maps


def build_symmetric_from_spec(
    components: Sequence[dict[str, Any]],
    control_maps: Sequence[list[int]],
) -> tuple[
    list[ComponentType],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
]:
    """Build all parameters for symmetric structure from specification.

    Args:
        components: List of factor specifications (same format as build_matrices_from_spec)
        control_maps: Control maps for each factor. control_maps[i] should have
            shape [prod(V_j for j!=i)] mapping other-factor tokens to variant index.

    Returns:
        Tuple of (component_types, transition_matrices, normalizing_eigenvectors,
                 initial_states, control_maps_arrays)

    Example:
        ```python
        components = [
            {
                "component_type": "hmm",
                "variants": [
                    {"process_name": "mess3", "x": 0.15, "a": 0.6},
                    {"process_name": "mess3", "x": 0.5, "a": 0.6},
                ],
            },
            # ... more components
        ]
        control_maps = [
            [0, 1, 0, 1],  # Factor 0: 4 other-token combos -> variants
            [1, 0, 1, 0],  # Factor 1: 4 other-token combos -> variants
        ]
        ```
    """
    # Build base matrices
    component_types, transition_matrices, normalizing_eigenvectors, initial_states = build_matrices_from_spec(
        components
    )

    # Convert control maps to JAX arrays
    control_maps_arrays = [jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps]

    # Validate control map lengths
    vocab_sizes = [int(T.shape[1]) for T in transition_matrices]
    F = len(vocab_sizes)

    for i, cm in enumerate(control_maps_arrays):
        # Expected length: product of all vocab sizes except i
        expected = 1
        for j in range(F):
            if j != i:
                expected *= vocab_sizes[j]

        if int(cm.shape[0]) != expected:
            raise ValueError(f"control_maps[{i}] length {cm.shape[0]} must equal prod(V_j for j!=[{i}]) = {expected}")

    return component_types, transition_matrices, normalizing_eigenvectors, initial_states, control_maps_arrays


def build_transition_coupled_from_spec(
    components: Sequence[dict[str, Any]],
    control_maps_transition: Sequence[list[int]],
    emission_variant_indices: Sequence[int],
    emission_control_maps: Sequence[list[int] | None] | None = None,
) -> tuple[
    list[ComponentType],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    jax.Array,
    list[jax.Array | None] | None,
]:
    """Build all parameters for transition-coupled structure from specification.

    Args:
        components: List of factor specifications
        control_maps_transition: Transition control maps (same format as symmetric)
        emission_variant_indices: Fixed emission variant per factor
        emission_control_maps: Optional chain-style emission control maps

    Returns:
        Tuple of (component_types, transition_matrices, normalizing_eigenvectors,
                 initial_states, control_maps_transition_arrays,
                 emission_variant_indices_array, emission_control_maps_arrays)

    Example:
        ```python
        components = [...]
        control_maps_transition = [[0, 1, 0, 1], [1, 0, 1, 0]]
        emission_variant_indices = [0, 0]  # Use variant 0 for emissions
        emission_control_maps = None  # Independent emissions
        ```
    """
    # Build base matrices
    component_types, transition_matrices, normalizing_eigenvectors, initial_states = build_matrices_from_spec(
        components
    )

    # Convert transition control maps
    control_maps_arrays = [jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps_transition]

    # Convert emission variant indices
    emission_variant_indices_array = jnp.asarray(emission_variant_indices, dtype=jnp.int32)

    # Convert emission control maps if provided
    emission_control_maps_arrays = None
    if emission_control_maps is not None:
        emission_control_maps_arrays = [
            jnp.asarray(cm, dtype=jnp.int32) if cm is not None else None for cm in emission_control_maps
        ]

    return (
        component_types,
        transition_matrices,
        normalizing_eigenvectors,
        initial_states,
        control_maps_arrays,
        emission_variant_indices_array,
        emission_control_maps_arrays,
    )


def _build_components_from_spec(
    components: Sequence[dict[str, Any]],
    device: str | None = None,
) -> list[GenerativeProcess]:
    """Build component GenerativeProcess instances from specifications.

    Args:
        components: List of component specs. Each spec has:
            - component_type: "hmm", "ghmm", or "factored"
            - For hmm/ghmm: process_name, process_params
            - For factored: structure_type, spec, and structure-specific params
        device: Device placement.

    Returns:
        List of built GenerativeProcess instances.

    Raises:
        ValueError: If component_type is unknown.
    """
    built_components = []

    for comp_spec in components:
        comp_type = comp_spec.get("component_type", "hmm")

        if comp_type == "hmm":
            process: GenerativeProcess = build_hidden_markov_model(
                process_name=comp_spec["process_name"],
                process_params=comp_spec.get("process_params", {}),
                device=device,
            )
        elif comp_type == "ghmm":
            process = build_generalized_hidden_markov_model(
                process_name=comp_spec["process_name"],
                process_params=comp_spec.get("process_params", {}),
                device=device,
            )
        elif comp_type == "factored":
            factored_kwargs = {k: v for k, v in comp_spec.items() if k not in ("component_type", "vocab_map")}
            process = build_factored_process_from_spec(**factored_kwargs)
        else:
            raise ValueError(f"Unknown component_type: {comp_type}")

        built_components.append(process)

    return built_components


def build_nonergodic_process_from_spec(
    components: Sequence[dict[str, Any]],
    component_weights: Sequence[float],
    vocab_maps: Sequence[Sequence[int]] | None = None,
    device: str | None = None,
) -> NonErgodicGenerativeProcess:
    """Build a nonergodic process from component specifications.

    Creates a NonErgodicGenerativeProcess that composes multiple GenerativeProcess
    instances into a truly nonergodic mixture with block diagonal structure.

    Args:
        components: List of component specs. Each spec has:
            - component_type: "hmm", "ghmm", or "factored"
            - For hmm/ghmm: process_name, process_params
            - For factored: structure_type, spec, and structure-specific params
            - vocab_map: Optional per-component vocab mapping
        component_weights: Mixture weights for components (will be normalized).
        vocab_maps: Optional global vocab maps (overrides per-component).
        device: Device placement.

    Returns:
        NonErgodicGenerativeProcess instance.

    Example:
        ```yaml
        instance:
          _target_: simplexity.generative_processes.builder.build_nonergodic_process_from_spec
          components:
            - component_type: hmm
              process_name: mess3
              process_params: {x: 0.15, a: 0.6}
            - component_type: ghmm
              process_name: tom_quantum
              process_params: {alpha: 1.0, beta: 4.0}
            - component_type: factored
              structure_type: independent
              spec:
                - component_type: hmm
                  variants:
                    - process_name: coin
                      process_params: {p: 0.5}
          component_weights: [0.5, 0.3, 0.2]
          vocab_maps:
            - [0, 1, 2]
            - [0, 1, 2]
            - [0, 1]
        ```

    Raises:
        ValueError: If component_type is unknown.
    """
    built_components = _build_components_from_spec(components, device=device)

    if vocab_maps is None:
        inferred_vocab_maps = []
        for comp_spec, process in zip(components, built_components, strict=True):
            comp_vocab_map = comp_spec.get("vocab_map", list(range(process.vocab_size)))
            inferred_vocab_maps.append(comp_vocab_map)
        final_vocab_maps: Sequence[Sequence[int]] = inferred_vocab_maps
    else:
        final_vocab_maps = vocab_maps

    return NonErgodicGenerativeProcess(
        components=built_components,
        component_weights=component_weights,
        vocab_maps=final_vocab_maps,
        device=device,
    )


def build_nonergodic_disjoint_vocab(
    components: Sequence[dict[str, Any]],
    component_weights: Sequence[float],
    device: str | None = None,
) -> NonErgodicGenerativeProcess:
    """Build a nonergodic process where each component has a fully disjoint alphabet.

    Builds each component once to discover its vocab_size, then assigns
    non-overlapping vocab_maps: C0 -> [0..V0-1], C1 -> [V0..V0+V1-1], etc.

    Args:
        components: List of component specs (same format as build_nonergodic_process_from_spec).
        component_weights: Mixture weights for components.
        device: Device placement.

    Returns:
        NonErgodicGenerativeProcess with disjoint per-component vocabularies.
    """
    built_components = _build_components_from_spec(components, device=device)

    vocab_maps: list[list[int]] = []
    offset = 0
    for c in built_components:
        vocab_maps.append(list(range(offset, offset + c.vocab_size)))
        offset += c.vocab_size

    return NonErgodicGenerativeProcess(
        components=built_components,
        component_weights=component_weights,
        vocab_maps=vocab_maps,
        device=device,
    )


def _build_prefix_vocab_maps(n_components: int, v: int, n_shared: int, n_unique: int) -> list[list[int]]:
    """Build vocab maps using the prefix strategy.

    C0 gets [0..V-1]. Ci>0 gets shared [0..n_shared-1] + unique tokens above V.
    """
    return [list(range(v))] + [
        list(range(n_shared)) + list(range(v + i * n_unique, v + (i + 1) * n_unique)) for i in range(n_components - 1)
    ]


def _build_sliding_vocab_maps(n_components: int, v: int, n_unique: int) -> list[list[int]]:
    """Build vocab maps using the sliding/offset strategy.

    Ci gets [i*offset..i*offset+V-1] where offset = max(1, n_unique).
    """
    offset = max(1, n_unique)
    return [list(range(i * offset, i * offset + v)) for i in range(n_components)]


def _build_random_vocab_maps(n_components: int, v: int, n_unique: int, seed: int) -> list[list[int]]:
    """Build vocab maps by having each component randomly sample V tokens from the global pool.

    The global vocab size is the same as in prefix mode:
    V + (n_components - 1) * n_unique.
    """
    global_vocab_size = v + (n_components - 1) * n_unique
    rng = random.Random(seed)
    return [sorted(rng.sample(range(global_vocab_size), v)) for _ in range(n_components)]


def build_nonergodic_partial_overlap(
    components: Sequence[dict[str, Any]],
    component_weights: Sequence[float],
    overlap_frac: float = 0.7,
    mode: Literal["prefix", "sliding", "random"] = "prefix",
    seed: int | None = None,
    device: str | None = None,
) -> NonErgodicGenerativeProcess:
    """Build a nonergodic process with partially overlapping alphabets.

    Args:
        components: List of component specs (same format as build_nonergodic_process_from_spec).
        component_weights: Mixture weights for components.
        overlap_frac: Fraction of tokens shared between components (0.0 = disjoint, 1.0 = full overlap).
        mode: Strategy for assigning vocab maps:
            - "prefix": C0 gets [0..V-1], Ci>0 gets shared prefix + unique suffix above V.
            - "sliding": Each component's vocab is offset by V * (1 - overlap_frac) from the previous.
            - "random": Each component independently samples V tokens from the global pool.
              Global pool size matches prefix mode. Requires the ``seed`` parameter.
        seed: Random seed for reproducibility. Required when mode="random".
        device: Device placement.

    Returns:
        NonErgodicGenerativeProcess with partially overlapping vocabularies.

    Raises:
        ValueError: If mode is unknown or seed is missing for random mode.
    """
    if mode == "random" and seed is None:
        raise ValueError("seed is required when mode='random'")

    built_components = _build_components_from_spec(components, device=device)
    comp_vocab_sizes = [c.vocab_size for c in built_components]
    if len(set(comp_vocab_sizes)) != 1:
        raise ValueError(f"All components must have equal vocab_size for partial_overlap, got {comp_vocab_sizes}")
    v = comp_vocab_sizes[0]
    n_shared = int(v * overlap_frac)
    n_unique = v - n_shared
    n_components = len(components)

    if mode == "prefix":
        vocab_maps = _build_prefix_vocab_maps(n_components, v, n_shared, n_unique)
    elif mode == "sliding":
        vocab_maps = _build_sliding_vocab_maps(n_components, v, n_unique)
    elif mode == "random":
        if seed is None:
            raise ValueError("seed is required when mode='random'")
        vocab_maps = _build_random_vocab_maps(n_components, v, n_unique, seed)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'prefix', 'sliding', or 'random'.")

    return NonErgodicGenerativeProcess(
        components=built_components,
        component_weights=component_weights,
        vocab_maps=vocab_maps,
        device=device,
    )


def build_inflated_process(
    base_process: GenerativeProcess,
    inflation_factor: int,
) -> InflatedVocabularyProcess:
    """Build an inflated vocabulary process wrapping a base process.

    Args:
        base_process: Any GenerativeProcess to wrap.
        inflation_factor: Number of noise variants per base token (K >= 2).

    Returns:
        InflatedVocabularyProcess with vocab_size = K * base_process.vocab_size.
    """
    return InflatedVocabularyProcess(base_process, inflation_factor)


def build_inflated_process_from_spec(
    base_spec: dict[str, Any],
    inflation_factor: int,
    device: str | None = None,
) -> InflatedVocabularyProcess:
    """Build an inflated vocabulary process from a base process specification.

    Args:
        base_spec: Specification for the base process. Must include:
            - component_type: "hmm", "ghmm", or "factored"
            - For hmm/ghmm: process_name, process_params
            - For factored: structure_type, spec, and structure-specific params
        inflation_factor: Number of noise variants per base token (K >= 2).
        device: Device placement.

    Returns:
        InflatedVocabularyProcess wrapping the built base process.

    Raises:
        ValueError: If component_type is unknown.
    """
    comp_type = base_spec.get("component_type", "hmm")

    if comp_type == "hmm":
        base_process: GenerativeProcess = build_hidden_markov_model(
            process_name=base_spec["process_name"],
            process_params=base_spec.get("process_params", {}),
            device=device,
            noise_epsilon=base_spec.get("noise_epsilon", 0.0),
        )
    elif comp_type == "ghmm":
        base_process = build_generalized_hidden_markov_model(
            process_name=base_spec["process_name"],
            process_params=base_spec.get("process_params", {}),
            device=device,
            noise_epsilon=base_spec.get("noise_epsilon", 0.0),
        )
    elif comp_type == "factored":
        factored_kwargs = {k: v for k, v in base_spec.items() if k not in ("component_type",)}
        base_process = build_factored_process_from_spec(**factored_kwargs)
    else:
        raise ValueError(f"Unknown base component_type: {comp_type}")

    return InflatedVocabularyProcess(base_process, inflation_factor)
