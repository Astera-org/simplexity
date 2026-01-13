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
    - Concatenated: "concatenated" → "Lcat"
    - Top-level hooks: "hook_embed" → "embed", "hook_pos_embed" → "pos_embed"
    - Block direct hooks: "blocks.N.hook_X_Y" → "LN.X.Y"
    - Block component hooks: "blocks.N.{comp}.hook_X" → "LN.{comp}.X"
    - ln_final hooks: "ln_final.hook_X" → "ln_final.X"
    - Other layers: unchanged

    Args:
        layer_name: Original layer name from activations dict

    Returns:
        Formatted layer spec

    Examples:
        >>> format_layer_spec("blocks.2.hook_resid_post")
        "L2.resid.post"
        >>> format_layer_spec("blocks.0.attn.hook_q")
        "L0.attn.q"
        >>> format_layer_spec("hook_embed")
        "embed"
        >>> format_layer_spec("ln_final.hook_scale")
        "ln_final.scale"
        >>> format_layer_spec("concatenated")
        "Lcat"
    """
    if layer_name == "concatenated":
        return "Lcat"

    if layer_name.startswith("hook_"):
        return layer_name[5:]

    ln_final_pattern = r"^ln_final\.hook_(?P<hook_name>.+)$"
    match = re.match(ln_final_pattern, layer_name)
    if match:
        return f"ln_final.{match.group('hook_name')}"

    if not layer_name.startswith("blocks."):
        return layer_name

    direct_hook_pattern = r"^blocks\.(?P<block_num>\d+)\.hook_(?P<hook_name>.+)$"
    match = re.match(direct_hook_pattern, layer_name)
    if match:
        block_num = match.group("block_num")
        hook_name = match.group("hook_name").replace("_", ".")
        return f"L{block_num}.{hook_name}"

    component_hook_pattern = r"^blocks\.(?P<block_num>\d+)\.(?P<component>\w+)\.hook_(?P<hook_name>.+)$"
    match = re.match(component_hook_pattern, layer_name)
    if match:
        block_num = match.group("block_num")
        component = match.group("component")
        hook_name = match.group("hook_name").replace("_", ".")
        return f"L{block_num}.{component}.{hook_name}"

    return layer_name
