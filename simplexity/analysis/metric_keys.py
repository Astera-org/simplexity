"""Utility functions for constructing layer-specific analysis keys for scalar and array metrics."""

from __future__ import annotations

import re


def construct_layer_specific_key(key: str, layer_name: str) -> str:
    """Construct a layer-specific namespaced metric key."""
    if "/" not in key:
        return f"{key}/{layer_name}"

    # If the key is factor-specific (e.g. "rmse/F0")
    # prepend the layer name to the factor (e.g. "rmse/L0.resid.post-F0")
    analysis, factor = key.rsplit("/", 1)
    if factor.startswith("F"):
        return f"{analysis}/{layer_name}-{factor}"

    return f"{key}/{layer_name}"


def format_layer_spec(layer_name: str) -> str:
    """Format layer name into compact layer specification.

    Converts verbose layer names to compact specs:
    - Block layers: "blocks.N.hook_X_Y" → "LN.X.Y"
    - Special layers: "embed", "pos_embed", "ln_final" → unchanged
    - Concatenated: "concatenated" → "Lcat"

    Args:
        layer_name: Original layer name from activations dict

    Returns:
        Formatted layer spec

    Examples:
        >>> format_layer_spec("blocks.2.hook_resid_post")
        "L2.resid.post"
        >>> format_layer_spec("blocks.0.hook_resid_pre")
        "L0.resid.pre"
        >>> format_layer_spec("blocks.10.hook_mlp_out")
        "L10.mlp.out"
        >>> format_layer_spec("embed")
        "embed"
        >>> format_layer_spec("concatenated")
        "Lcat"
    """
    if layer_name == "concatenated":
        return "Lcat"

    if not layer_name.startswith("blocks."):
        return layer_name

    block_pattern = r"^blocks\.(?P<block_num>\d+)\.hook_(?P<hook_name>.+)$"
    match = re.match(block_pattern, layer_name)
    if match:
        block_num = match.group("block_num")
        hook_name = match.group("hook_name")
        simplified_hook_name = hook_name.replace("_", ".")
        return f"L{block_num}.{simplified_hook_name}"

    return layer_name
