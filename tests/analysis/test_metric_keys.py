"""Tests for the metric key construction utility functions."""

from simplexity.analysis.metric_keys import construct_layer_specific_key, format_layer_spec


def test_construct_layer_specific_key_given_factor_specific_key() -> None:
    """Test that the function adds layer name before the factor-specific key."""
    key = "rmse/F0"
    layer_name = "L1.resid.post"
    expected_key = "rmse/L1.resid.post-F0"
    assert construct_layer_specific_key(key, layer_name) == expected_key


def test_construct_layer_specific_key_given_non_factor_specific_key() -> None:
    """Test that the function adds layer name before the non-factor-specific key."""
    key = "r2"
    layer_name = "L1.resid.post"
    expected_key = "r2/L1.resid.post"
    assert construct_layer_specific_key(key, layer_name) == expected_key


def test_construct_layer_specific_key_given_non_factor_suffix() -> None:
    """Test that a key with '/' but a non-F suffix appends the layer name."""
    key = "orth/overlap"
    layer_name = "L0.resid.post"
    expected_key = "orth/overlap/L0.resid.post"
    assert construct_layer_specific_key(key, layer_name) == expected_key


def test_format_layer_spec_concatenated() -> None:
    """Test that the function returns the correct format for concatenated layers."""
    layer_name = "concatenated"
    expected_key = "Lcat"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_block_and_hook_layer() -> None:
    """Test that the function returns the correct format for block and hook layer name."""
    layer_name = "blocks.2.hook_resid_post"
    expected_key = "L2.resid.post"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_special_layer() -> None:
    """Test that the function returns the correct format for special layer name."""
    layer_name = "embed"
    expected_key = "embed"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_block_layer_with_no_hook_name() -> None:
    """Test that the function returns the input layer name if it is a block layer name with no hook name."""
    layer_name = "blocks.2"
    expected_key = "blocks.2"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_block_and_hook_layer_with_no_block_number() -> None:
    """Test that the function returns the input layer name if it is a block and hook layer name with no block number."""
    layer_name = "blocks.hook_resid_post"
    expected_key = "blocks.hook_resid_post"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_block_and_hook_layer_with_extra_structure() -> None:
    """Test that the function returns the correct format if it is a block and hook layer name with extra structure."""
    layer_name = "blocks.2.hook_resid_post.invalid"
    expected_key = "L2.resid.post.invalid"
    assert format_layer_spec(layer_name) == expected_key


def test_format_layer_spec_hook_embed() -> None:
    """Test that hook_embed is formatted to embed."""
    assert format_layer_spec("hook_embed") == "embed"


def test_format_layer_spec_hook_pos_embed() -> None:
    """Test that hook_pos_embed is formatted to pos_embed."""
    assert format_layer_spec("hook_pos_embed") == "pos_embed"


def test_format_layer_spec_block_component_attn_hook() -> None:
    """Test that block attention component hooks are formatted correctly."""
    assert format_layer_spec("blocks.0.attn.hook_q") == "L0.attn.q"
    assert format_layer_spec("blocks.1.attn.hook_k") == "L1.attn.k"
    assert format_layer_spec("blocks.2.attn.hook_v") == "L2.attn.v"
    assert format_layer_spec("blocks.0.attn.hook_attn_scores") == "L0.attn.attn.scores"
    assert format_layer_spec("blocks.0.attn.hook_pattern") == "L0.attn.pattern"
    assert format_layer_spec("blocks.0.attn.hook_z") == "L0.attn.z"


def test_format_layer_spec_block_component_ln_hook() -> None:
    """Test that block layer norm hooks are formatted correctly."""
    assert format_layer_spec("blocks.0.ln1.hook_scale") == "L0.ln1.scale"
    assert format_layer_spec("blocks.0.ln1.hook_normalized") == "L0.ln1.normalized"
    assert format_layer_spec("blocks.1.ln2.hook_scale") == "L1.ln2.scale"


def test_format_layer_spec_block_component_mlp_hook() -> None:
    """Test that block MLP hooks are formatted correctly."""
    assert format_layer_spec("blocks.0.mlp.hook_pre") == "L0.mlp.pre"
    assert format_layer_spec("blocks.0.mlp.hook_post") == "L0.mlp.post"


def test_format_layer_spec_ln_final_hook() -> None:
    """Test that ln_final hooks are formatted correctly."""
    assert format_layer_spec("ln_final.hook_scale") == "ln_final.scale"
    assert format_layer_spec("ln_final.hook_normalized") == "ln_final.normalized"


def test_format_layer_spec_ln_final_passthrough() -> None:
    """Test that ln_final without hook prefix is passed through unchanged."""
    assert format_layer_spec("ln_final") == "ln_final"
