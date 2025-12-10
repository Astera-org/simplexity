"""Pattern parsing and expansion logic for visualization field mappings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import numpy as np

from simplexity.activations.visualization.field_resolution import _lookup_projection_array
from simplexity.activations.visualization.pattern_utils import (
    build_wildcard_regex,
    count_patterns,
    has_pattern,
    parse_range,
    substitute_pattern,
    validate_single_pattern,
)
from simplexity.activations.visualization_configs import ActivationVisualizationFieldRef
from simplexity.exceptions import ConfigValidationError


def _has_key_pattern(key: str | None) -> bool:
    """Check if key contains * or range pattern (e.g., factor_*/projected)."""
    if key is None:
        return False
    validate_single_pattern(key, "Key")
    return has_pattern(key)


def _has_field_pattern(field_name: str) -> bool:
    """Check if field name contains * or range pattern."""
    validate_single_pattern(field_name, "Field name")
    return has_pattern(field_name)


def _parse_component_spec(component: int | str | None) -> tuple[str, int | None, int | None]:
    """Parse component into (type, start, end).

    Returns:
        - ("single", val, None) for int component
        - ("wildcard", None, None) for "*"
        - ("range", start, end) for "start...end"
        - ("none", None, None) for None
    """
    if component is None:
        return ("none", None, None)
    if isinstance(component, int):
        return ("single", component, None)
    if component == "*":
        return ("wildcard", None, None)
    if "..." in component:
        parts = component.split("...")
        if len(parts) != 2:
            raise ConfigValidationError(f"Invalid range: {component}")
        try:
            start, end = int(parts[0]), int(parts[1])
            if start >= end:
                raise ConfigValidationError(f"Range start must be < end: {component}")
            return ("range", start, end)
        except ValueError as e:
            raise ConfigValidationError(f"Invalid range: {component}") from e
    raise ConfigValidationError(f"Unrecognized component pattern: {component}")


def _expand_pattern_to_indices(
    pattern: str,
    available_keys: Iterable[str],
) -> list[int]:
    """Extract numeric indices from keys matching a wildcard or range pattern.

    Args:
        pattern: Pattern with * or N...M
        available_keys: Keys to match against

    Returns:
        Sorted list of unique indices that match the pattern
    """
    if not has_pattern(pattern):
        raise ConfigValidationError(f"Pattern '{pattern}' has no wildcard or range")

    if "*" in pattern:
        regex_pattern = build_wildcard_regex(pattern)
        indices: list[int] = []
        for key in available_keys:
            match = regex_pattern.match(key)
            if match:
                try:
                    indices.append(int(match.group(1)))
                except (ValueError, IndexError):
                    continue
        if not indices:
            raise ConfigValidationError(f"No keys found matching pattern '{pattern}'")
        return sorted(set(indices))
    else:
        range_bounds = parse_range(pattern)
        if not range_bounds:
            raise ConfigValidationError(f"Invalid range pattern in '{pattern}'")
        start_idx, end_idx = range_bounds
        return list(range(start_idx, end_idx))


def _get_component_count(
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
) -> int:
    """Get number of components available for expansion."""
    if ref.source == "projections":
        if ref.key is None:
            raise ConfigValidationError("Projection refs require key")
        array = _lookup_projection_array(projections, layer_name, ref.key, analysis_concat_layers)
        np_array = np.asarray(array)
        if np_array.ndim == 1:
            raise ConfigValidationError(f"Cannot expand 1D projection '{ref.key}'. Patterns require 2D arrays.")
        if np_array.ndim != 2:
            raise ConfigValidationError(f"Projection must be 1D or 2D, got {np_array.ndim}D")
        return np_array.shape[1]

    elif ref.source == "belief_states":
        if belief_states is None:
            raise ConfigValidationError("Belief states not available")
        np_array = np.asarray(belief_states)
        if np_array.ndim != 2:
            raise ConfigValidationError(f"Belief states must be 2D, got {np_array.ndim}D")
        return np_array.shape[1]

    else:
        raise ConfigValidationError(f"Component expansion not supported for source: {ref.source}")


def _expand_projection_key_pattern(
    key_pattern: str,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    analysis_concat_layers: bool,
) -> dict[str, str]:
    """Expand projection key patterns against available keys.

    Args:
        key_pattern: Pattern like "factor_*/projected" or "factor_0...3/projected"
        layer_name: Current layer name for matching
        projections: Available projection arrays
        analysis_concat_layers: Whether layers were concatenated

    Returns:
        Dict mapping extracted index (as string) to the concrete key suffix.
        E.g., {"0": "factor_0/projected", "1": "factor_1/projected"}
    """
    # Build regex from pattern
    if "*" in key_pattern:
        regex_pattern = build_wildcard_regex(key_pattern)
    else:
        # Range pattern like "factor_0...3/projected"
        range_bounds = parse_range(key_pattern)
        if not range_bounds:
            raise ConfigValidationError(f"Invalid key pattern: {key_pattern}")
        start_idx, end_idx = range_bounds
        if start_idx >= end_idx:
            raise ConfigValidationError(f"Invalid range in key pattern: {key_pattern}")
        # Return explicit range without matching
        result = {}
        for idx in range(start_idx, end_idx):
            concrete_key = substitute_pattern(key_pattern, idx)
            result[str(idx)] = concrete_key
        return result

    # Match against available projection keys
    result: dict[str, str] = {}
    for full_key in projections:
        # Extract the key suffix (part after layer name)
        if analysis_concat_layers:
            # Keys are like "factor_0/projected" directly
            key_suffix = full_key
        else:
            # Keys are like "layer_name_factor_0/projected"
            prefix = f"{layer_name}_"
            if not full_key.startswith(prefix):
                continue
            key_suffix = full_key[len(prefix) :]

        match = regex_pattern.match(key_suffix)
        if match:
            extracted_idx = match.group(1)
            if extracted_idx not in result:
                result[extracted_idx] = key_suffix

    if not result:
        raise ConfigValidationError(
            f"No projection keys found matching pattern '{key_pattern}' for layer '{layer_name}'. "
            f"Available keys: {list(projections.keys())}"
        )

    return result


def _expand_projection_key_mapping(
    field_name: str,
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
) -> dict[str, ActivationVisualizationFieldRef]:
    """Expand projection key patterns, optionally combined with component patterns.

    Handles cross-product expansion when both key and component patterns are present.
    Sets _group_value on expanded refs for DataFrame construction.
    """
    assert ref.key is not None, "Key must be provided for projection key pattern expansion"

    # Expand key pattern to get concrete keys
    key_expansions = _expand_projection_key_pattern(ref.key, layer_name, projections, analysis_concat_layers)

    # Check if component expansion is also needed
    spec_type, start_idx, end_idx = _parse_component_spec(ref.component)
    needs_component_expansion = spec_type in ("wildcard", "range")

    expanded: dict[str, ActivationVisualizationFieldRef] = {}

    # Count patterns in field name to handle cross-product correctly
    total_field_patterns = count_patterns(field_name)

    for group_idx, concrete_key in sorted(key_expansions.items(), key=lambda x: int(x[0])):
        if needs_component_expansion:
            # Get component count for this specific key
            array = _lookup_projection_array(projections, layer_name, concrete_key, analysis_concat_layers)
            np_array = np.asarray(array)
            if np_array.ndim != 2:
                raise ConfigValidationError(
                    f"Component expansion requires 2D projection, got {np_array.ndim}D for key '{concrete_key}'"
                )
            max_components = np_array.shape[1]

            if spec_type == "wildcard":
                components = list(range(max_components))
            else:
                assert start_idx is not None
                assert end_idx is not None
                if end_idx > max_components:
                    raise ConfigValidationError(
                        f"Range {start_idx}...{end_idx} exceeds components ({max_components}) for key '{concrete_key}'"
                    )
                components = list(range(start_idx, end_idx))

            # Cross-product: expand both key and component
            for comp_idx in components:
                # Replace patterns in field name (key pattern first, then component)
                if total_field_patterns == 2:
                    # Two patterns: first for key, second for component
                    expanded_name = substitute_pattern(field_name, int(group_idx))
                    expanded_name = substitute_pattern(expanded_name, comp_idx)
                elif total_field_patterns == 1:
                    # Only one pattern in field name - use for component, append group index
                    expanded_name = substitute_pattern(field_name, comp_idx)
                else:
                    raise ConfigValidationError(
                        f"Field '{field_name}' must have 1-2 patterns for key+component expansion"
                    )

                expanded[expanded_name] = ActivationVisualizationFieldRef(
                    source="projections",
                    key=concrete_key,
                    component=comp_idx,
                    reducer=ref.reducer,
                    group_as=ref.group_as,
                    _group_value=str(group_idx),
                )
        else:
            # Only key pattern, no component expansion
            expanded_name = substitute_pattern(field_name, int(group_idx))

            expanded[expanded_name] = ActivationVisualizationFieldRef(
                source="projections",
                key=concrete_key,
                component=ref.component,  # Keep original (could be None or int)
                reducer=ref.reducer,
                group_as=ref.group_as,
                _group_value=str(group_idx),
            )

    return expanded


def _expand_belief_factor_mapping(
    field_name: str,
    ref: ActivationVisualizationFieldRef,
    belief_states: np.ndarray,
) -> dict[str, ActivationVisualizationFieldRef]:
    """Expand belief state factor patterns, optionally combined with component patterns.

    Handles cross-product expansion when both factor and component patterns are present.
    Sets _group_value on expanded refs for DataFrame construction.
    """
    np_beliefs = np.asarray(belief_states)
    if np_beliefs.ndim != 3:
        raise ConfigValidationError(
            f"Belief state factor patterns require 3D beliefs (samples, factors, states), got {np_beliefs.ndim}D"
        )

    n_factors = np_beliefs.shape[1]
    n_states = np_beliefs.shape[2]

    # Parse factor pattern
    factor_spec = ref.factor
    assert isinstance(factor_spec, str), "Factor should be a pattern string"

    if factor_spec == "*":
        factors = list(range(n_factors))
    elif "..." in factor_spec:
        parts = factor_spec.split("...")
        start_idx, end_idx = int(parts[0]), int(parts[1])
        if end_idx > n_factors:
            raise ConfigValidationError(f"Factor range {start_idx}...{end_idx} exceeds available factors ({n_factors})")
        factors = list(range(start_idx, end_idx))
    else:
        raise ConfigValidationError(f"Invalid factor pattern: {factor_spec}")

    # Check if component expansion is also needed
    spec_type, start_idx, end_idx = _parse_component_spec(ref.component)
    needs_component_expansion = spec_type in ("wildcard", "range")

    expanded: dict[str, ActivationVisualizationFieldRef] = {}

    # Count patterns in field name
    total_field_patterns = count_patterns(field_name)

    for factor_idx in factors:
        if needs_component_expansion:
            # Get component range
            if spec_type == "wildcard":
                components = list(range(n_states))
            else:
                assert start_idx is not None
                assert end_idx is not None
                if end_idx > n_states:
                    raise ConfigValidationError(f"Component range {start_idx}...{end_idx} exceeds states ({n_states})")
                components = list(range(start_idx, end_idx))

            # Cross-product: expand both factor and component
            for comp_idx in components:
                if total_field_patterns == 2:
                    # Two patterns: first for factor, second for component
                    expanded_name = substitute_pattern(field_name, factor_idx)
                    expanded_name = substitute_pattern(expanded_name, comp_idx)
                elif total_field_patterns == 1:
                    # Only one pattern - use for component, append factor index
                    expanded_name = substitute_pattern(field_name, comp_idx)
                else:
                    raise ConfigValidationError(
                        f"Field '{field_name}' must have 1-2 patterns for factor+component expansion"
                    )

                expanded[expanded_name] = ActivationVisualizationFieldRef(
                    source="belief_states",
                    key=ref.key,
                    component=comp_idx,
                    reducer=ref.reducer,
                    group_as=ref.group_as,
                    factor=factor_idx,
                    _group_value=str(factor_idx),
                )
        else:
            # Only factor pattern, no component expansion
            expanded_name = substitute_pattern(field_name, factor_idx)

            expanded[expanded_name] = ActivationVisualizationFieldRef(
                source="belief_states",
                key=ref.key,
                component=ref.component,
                reducer=ref.reducer,
                group_as=ref.group_as,
                factor=factor_idx,
                _group_value=str(factor_idx),
            )

    return expanded


def _expand_scalar_keys(
    field_pattern: str,
    key_pattern: str | None,
    layer_name: str,
    scalars: Mapping[str, float],
) -> dict[str, str]:
    """Expand scalar field patterns by matching available scalar keys.

    Returns dict of expanded field_name → scalar_key.
    """
    if key_pattern is None:
        raise ConfigValidationError("Scalar wildcard expansion requires a key pattern")

    if not has_pattern(key_pattern):
        return {field_pattern: key_pattern}

    indices = _expand_pattern_to_indices(key_pattern, scalars.keys())

    expanded = {}
    for idx in indices:
        expanded_field = substitute_pattern(field_pattern, idx) if has_pattern(field_pattern) else field_pattern
        expanded_key = substitute_pattern(key_pattern, idx)
        expanded[expanded_field] = expanded_key

    return expanded


def _expand_scalar_pattern_keys(
    pattern: str,
    available_keys: Iterable[str],
    analysis_name: str,
) -> list[str]:
    """Expand wildcard/range pattern against available scalar keys."""
    keys = list(available_keys)
    has_prefixed_keys = any("/" in key for key in keys)
    prefix = f"{analysis_name}/"

    normalized_pattern = pattern
    if "/" not in normalized_pattern and has_prefixed_keys:
        normalized_pattern = f"{analysis_name}/{normalized_pattern}"
    elif "/" in normalized_pattern and not has_prefixed_keys and normalized_pattern.startswith(prefix):
        normalized_pattern = normalized_pattern[len(prefix) :]

    pattern_variants = _expand_scalar_pattern_ranges(normalized_pattern)
    matched: list[str] = []

    for variant in pattern_variants:
        if "*" in variant:
            escaped = re.escape(variant).replace(r"\*", r"([^/]+)")
            regex = re.compile(f"^{escaped}$")
            matched.extend(key for key in keys if regex.match(key))
        else:
            if variant in keys:
                matched.append(variant)

    unique_matches: list[str] = []
    seen: set[str] = set()
    for key in matched:
        if key not in seen:
            seen.add(key)
            unique_matches.append(key)

    if not unique_matches:
        raise ConfigValidationError(f"No scalar pattern keys found matching pattern '{pattern}'")

    return sorted(unique_matches)


def _expand_scalar_pattern_ranges(pattern: str) -> list[str]:
    """Expand numeric range tokens (e.g., 0...4) within a scalar pattern."""
    range_bounds = parse_range(pattern)
    if not range_bounds:
        return [pattern]

    start_idx, end_idx = range_bounds
    if start_idx >= end_idx:
        raise ConfigValidationError(f"Invalid range pattern in scalar pattern key '{pattern}'")

    expanded: list[str] = []
    for idx in range(start_idx, end_idx):
        replaced = substitute_pattern(pattern, idx)
        expanded.extend(_expand_scalar_pattern_ranges(replaced))
    return expanded


def _scalar_pattern_label(full_key: str) -> str:
    """Derive a categorical label for scalar pattern rows based on the key."""
    suffix = full_key.split("/", 1)[1] if "/" in full_key else full_key
    layer_match = re.search(r"(layer_\d+)", suffix)
    if layer_match:
        return layer_match.group(1)
    return suffix


def _expand_field_mapping(
    field_name: str,
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
) -> dict[str, ActivationVisualizationFieldRef]:
    """Expand pattern-based mapping into concrete mappings.

    Returns dict of expanded field_name → FieldRef with concrete component/key values.
    """
    # Check for projection key patterns FIRST (allows multiple field patterns for key+component)
    if ref.source == "projections" and ref.key and _has_key_pattern(ref.key):
        # For key pattern expansion, we allow up to 2 patterns in field name
        # (one for key expansion, one for component expansion)
        total_field_patterns = count_patterns(field_name)

        if total_field_patterns == 0:
            raise ConfigValidationError(f"Projection key pattern '{ref.key}' requires field name pattern")
        if total_field_patterns > 2:
            raise ConfigValidationError(
                f"Field name '{field_name}' has too many patterns (max 2 for key+component expansion)"
            )

        return _expand_projection_key_mapping(
            field_name, ref, layer_name, projections, belief_states, analysis_concat_layers
        )

    # Check for belief state factor patterns
    if ref.source == "belief_states" and ref.factor is not None and isinstance(ref.factor, str):
        has_factor_pattern = ref.factor == "*" or "..." in ref.factor
        if has_factor_pattern:
            if belief_states is None:
                raise ConfigValidationError("Belief state factor patterns require belief_states to be provided")
            total_field_patterns = count_patterns(field_name)

            if total_field_patterns == 0:
                raise ConfigValidationError(f"Belief state factor pattern '{ref.factor}' requires field name pattern")
            if total_field_patterns > 2:
                raise ConfigValidationError(
                    f"Field name '{field_name}' has too many patterns (max 2 for factor+component expansion)"
                )

            return _expand_belief_factor_mapping(field_name, ref, belief_states)

    field_has_pattern = _has_field_pattern(field_name)

    if ref.source == "scalars":
        key_has_pattern = ref.key is not None and has_pattern(ref.key)

        if field_has_pattern and not key_has_pattern:
            raise ConfigValidationError(f"Field '{field_name}' has pattern but scalar key has no pattern")
        if key_has_pattern and not field_has_pattern:
            raise ConfigValidationError(f"Scalar key pattern '{ref.key}' requires field name pattern")

        if not field_has_pattern:
            return {field_name: ref}

        scalar_expansions = _expand_scalar_keys(field_name, ref.key, layer_name, scalars)
        return {
            field: ActivationVisualizationFieldRef(source="scalars", key=key, component=None, reducer=None)
            for field, key in scalar_expansions.items()
        }

    spec_type, start_idx, end_idx = _parse_component_spec(ref.component)
    needs_expansion = spec_type in ("wildcard", "range")

    if field_has_pattern and not needs_expansion:
        raise ConfigValidationError(f"Field '{field_name}' has pattern but component is not wildcard/range")
    if needs_expansion and not field_has_pattern:
        raise ConfigValidationError(f"Component pattern '{ref.component}' requires field name pattern")

    if not needs_expansion:
        return {field_name: ref}

    max_components = _get_component_count(ref, layer_name, projections, belief_states, analysis_concat_layers)

    if spec_type == "wildcard":
        components = list(range(max_components))
    else:
        assert start_idx is not None, "Range spec must have start index"
        assert end_idx is not None, "Range spec must have end index"
        if end_idx > max_components:
            raise ConfigValidationError(
                f"Range {start_idx}...{end_idx} exceeds available components (max: {max_components})"
            )
        components = list(range(start_idx, end_idx))

    expanded = {}
    for comp_idx in components:
        expanded_name = substitute_pattern(field_name, comp_idx)

        expanded[expanded_name] = ActivationVisualizationFieldRef(
            source=ref.source,
            key=ref.key,
            component=comp_idx,
            reducer=ref.reducer,
        )

    return expanded


__all__ = [
    "_expand_belief_factor_mapping",
    "_expand_field_mapping",
    "_expand_pattern_to_indices",
    "_expand_projection_key_mapping",
    "_expand_projection_key_pattern",
    "_expand_scalar_keys",
    "_expand_scalar_pattern_keys",
    "_expand_scalar_pattern_ranges",
    "_get_component_count",
    "_has_field_pattern",
    "_has_key_pattern",
    "_parse_component_spec",
    "_scalar_pattern_label",
]
