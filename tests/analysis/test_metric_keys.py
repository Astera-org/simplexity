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
