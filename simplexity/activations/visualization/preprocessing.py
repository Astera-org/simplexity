"""Post-processing transforms for visualization DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simplexity.activations.visualization.pattern_utils import (
    build_wildcard_regex,
    has_pattern,
    parse_range,
    substitute_pattern,
)
from simplexity.activations.visualization_configs import ActivationVisualizationPreprocessStep
from simplexity.analysis.pca import compute_weighted_pca
from simplexity.exceptions import ConfigValidationError


def _expand_preprocessing_fields(field_patterns: list[str], available_columns: list[str]) -> list[str]:
    """Expand wildcard and range patterns in preprocessing field lists.

    Args:
        field_patterns: List of field names, may contain patterns like "belief_*" or "prob_0...3"
        available_columns: List of column names available in the DataFrame

    Returns:
        Expanded list of field names with patterns replaced by matching columns
    """
    expanded: list[str] = []
    for pattern in field_patterns:
        # Check if this is a pattern
        if has_pattern(pattern):
            # Extract the numeric pattern if it's a range
            range_bounds = parse_range(pattern)
            if range_bounds:
                start, end = range_bounds
                component_range = list(range(start, end))
                # Replace range pattern with each index
                for idx in component_range:
                    expanded_name = substitute_pattern(pattern, idx)
                    if expanded_name in available_columns:
                        expanded.append(expanded_name)
                    else:
                        raise ConfigValidationError(
                            f"Preprocessing pattern '{pattern}' expanded to '{expanded_name}' "
                            f"but column not found in DataFrame. "
                            f"Available columns: {', '.join(sorted(available_columns))}"
                        )
            elif "*" in pattern:
                # Wildcard pattern - find all matching columns
                regex = build_wildcard_regex(pattern)
                matches = []
                for col in available_columns:
                    match = regex.match(col)
                    if match:
                        # Extract the numeric part for sorting
                        try:
                            idx = int(match.group(1))
                            matches.append((idx, col))
                        except (IndexError, ValueError):
                            continue
                if not matches:
                    raise ConfigValidationError(
                        f"Preprocessing pattern '{pattern}' did not match any columns in DataFrame. "
                        f"Available columns: {', '.join(sorted(available_columns))}"
                    )
                # Sort by index and add column names
                matches.sort(key=lambda x: x[0])
                expanded.extend([col for _, col in matches])
            else:
                raise ConfigValidationError(f"Invalid preprocessing field pattern: {pattern}")
        else:
            # Not a pattern, just add as-is
            expanded.append(pattern)

    return expanded


def _apply_preprocessing(dataframe: pd.DataFrame, steps: list[ActivationVisualizationPreprocessStep]) -> pd.DataFrame:
    """Apply preprocessing steps to a DataFrame."""
    result = dataframe.copy()
    available_columns = list(result.columns)

    for step in steps:
        # Validate output_fields don't contain patterns
        for output_field in step.output_fields:
            if "*" in output_field or "..." in output_field:
                raise ConfigValidationError(
                    f"Preprocessing output_fields cannot contain patterns. Found: '{output_field}'"
                )

        # Expand input_fields patterns
        expanded_input_fields = _expand_preprocessing_fields(step.input_fields, available_columns)

        # Create a modified step with expanded fields
        expanded_step = ActivationVisualizationPreprocessStep(
            type=step.type, input_fields=expanded_input_fields, output_fields=step.output_fields
        )

        if step.type == "project_to_simplex":
            result = _project_to_simplex(result, expanded_step)
        elif step.type == "combine_rgb":
            result = _combine_rgb(result, expanded_step)
        else:  # pragma: no cover - defensive for future types
            raise ConfigValidationError(f"Unsupported preprocessing op '{step.type}'")

        # Update available columns for next step
        available_columns = list(result.columns)

    return result


def _project_to_simplex(dataframe: pd.DataFrame, step: ActivationVisualizationPreprocessStep) -> pd.DataFrame:
    """Project 3D probability coordinates to 2D simplex coordinates."""
    required = step.input_fields
    for column in required:
        if column not in dataframe:
            raise ConfigValidationError(
                f"Preprocessing step requires column '{column}' but it is missing from the dataframe."
            )
    p0, p1, p2 = (dataframe[col].astype(float) for col in required)
    x = p1 + 0.5 * p2
    y = (np.sqrt(3.0) / 2.0) * p2
    dataframe[step.output_fields[0]] = x
    dataframe[step.output_fields[1]] = y
    return dataframe


def _combine_rgb(dataframe: pd.DataFrame, step: ActivationVisualizationPreprocessStep) -> pd.DataFrame:
    """Combine input fields into RGB color values.

    Supports either:
    - 3 input fields: Directly map to R, G, B channels
    - >3 input fields: Project to 3D via PCA, then map to RGB
    """
    # ---- Validation ----
    # Note: input_fields have already been expanded by _expand_preprocessing_fields()
    # at this point, so we just validate the expanded result
    if len(step.output_fields) != 1:
        raise ConfigValidationError("combine_rgb requires exactly one output_field.")
    if len(step.input_fields) < 3:
        raise ConfigValidationError("combine_rgb requires at least three input_fields.")

    # Make sure all input columns exist
    for field in step.input_fields:
        if field not in dataframe:
            raise ConfigValidationError(f"combine_rgb requires column '{field}' but it is missing from the dataframe.")

    def _channel_to_int(series: pd.Series) -> pd.Series:
        return (series.clip(0.0, 1.0) * 255).round().astype(int)

    # ---- Case 1: exactly 3 inputs -> keep old behavior ----
    if len(step.input_fields) == 3:
        r, g, b = step.input_fields
        r_vals = _channel_to_int(pd.Series(dataframe[r]))
        g_vals = _channel_to_int(pd.Series(dataframe[g]))
        b_vals = _channel_to_int(pd.Series(dataframe[b]))

    # ---- Case 2: >3 inputs -> PCA to 3D, then map to RGB ----
    else:
        import jax.numpy as jnp

        # Stack the selected columns into an (n_samples, n_features) matrix
        X_np = dataframe[step.input_fields].to_numpy(dtype=float)
        X_jax = jnp.asarray(X_np)

        # Unweighted PCA (weights=None) to up to 3 components
        # We pass n_components=3, but compute_weighted_pca will cap it at min(n_samples, n_features)
        # via its own logic if you change it to allow that, or you can just pass None and slice.
        pca_res = compute_weighted_pca(
            X_jax,
            n_components=None,  # let it pick max_rank
            weights=None,
            center=True,
        )

        # Get projected coordinates, shape: (n_samples, k) where k = max_rank
        proj = np.asarray(pca_res["X_proj"])  # convert from jax.Array to numpy

        # Ensure we have 3 channels: take first 3 components, pad with zeros if fewer
        if proj.shape[1] >= 3:
            proj3 = proj[:, :3]
        else:
            # This is rare (happens when n_samples < 3). Pad extra dims with zeros.
            pad_width = 3 - proj.shape[1]
            proj3 = np.pad(proj, ((0, 0), (0, pad_width)), mode="constant")

        # Min-max normalize each component to [0, 1] across the dataset
        mins = proj3.min(axis=0)
        maxs = proj3.max(axis=0)
        ranges = maxs - mins
        # Avoid divide-by-zero: if range is 0, just leave that channel at 0.5
        ranges_safe = np.where(ranges > 0, ranges, 1.0)
        colors = (proj3 - mins) / ranges_safe
        colors[:, ranges == 0] = 0.5

        colors = np.clip(colors, 0.0, 1.0)

        # Turn into Series so we can reuse _channel_to_int
        r_vals = _channel_to_int(pd.Series(colors[:, 0], index=dataframe.index))
        g_vals = _channel_to_int(pd.Series(colors[:, 1], index=dataframe.index))
        b_vals = _channel_to_int(pd.Series(colors[:, 2], index=dataframe.index))

    # ---- Build hex color column ----
    dataframe[step.output_fields[0]] = [
        f"#{rv:02x}{gv:02x}{bv:02x}" for rv, gv, bv in zip(r_vals, g_vals, b_vals, strict=False)
    ]
    return dataframe


__all__ = [
    "_apply_preprocessing",
    "_combine_rgb",
    "_expand_preprocessing_fields",
    "_project_to_simplex",
]
