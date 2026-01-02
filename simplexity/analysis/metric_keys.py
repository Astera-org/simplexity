"""Utility functions for constructing layer-specific analysis keys for scalar and array metrics."""

from __future__ import annotations

import re


def construct_layer_specific_key(key: str, layer_name: str) -> str:
    """Construct a layer-specific namespaced metric key."""
    split_key = key.split("/")
    last_part = split_key[-1]
    if last_part.startswith("F") and len(split_key) > 1:  # If the key is a factor-specific key, prepend the layer name
        new_last_part = f"{layer_name}-{last_part}"
        new_key = "/".join(split_key[:-1] + [new_last_part])
    else:
        new_last_part = f"{layer_name}"
        new_key = "/".join(split_key + [new_last_part])
    return new_key


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

    block_pattern = r"^blocks\.(\d+)\.hook_(.+)$"
    match = re.match(block_pattern, layer_name)
    if match:
        block_num = match.group(1)
        hook_name = match.group(2)
        simplified_hook_name = hook_name.replace("_", ".")
        return f"L{block_num}.{simplified_hook_name}"

    return layer_name
