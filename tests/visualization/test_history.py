"""Tests for visualization history persistence utilities."""

from __future__ import annotations

import copy

import pandas as pd

from simplexity.visualization.history import (
    history_paths,
    load_history_dataframe,
    plot_config_signature,
    save_history_dataframe,
)
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    DataConfig,
    GeometryConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)


def _simple_plot_config() -> PlotConfig:
    layer = LayerConfig(
        geometry=GeometryConfig(type="line"),
        aesthetics=AestheticsConfig(
            x=ChannelAestheticsConfig(field="step", type="quantitative"),
            y=ChannelAestheticsConfig(field="value", type="quantitative"),
        ),
    )
    return PlotConfig(
        backend="altair",
        data=DataConfig(source="main"),
        layers=[layer],
        size=PlotSizeConfig(width=400, height=200),
        guides=PlotLevelGuideConfig(),
    )


def test_plot_config_signature_changes_with_config_mutation():
    """Test that plot config signature changes when config is mutated."""
    cfg = _simple_plot_config()
    clone = copy.deepcopy(cfg)
    clone.size.width = 800

    assert plot_config_signature(cfg) != plot_config_signature(clone)


def test_history_round_trip(tmp_path):
    """Test saving and loading history dataframe preserves data."""
    cfg = _simple_plot_config()
    signature = plot_config_signature(cfg)
    data_path, meta_path = history_paths(tmp_path, "demo")
    df = pd.DataFrame({"step": [0, 1], "value": [0.1, 0.2]})

    save_history_dataframe(
        df,
        data_path,
        meta_path,
        signature=signature,
        analysis="analysis",
        name="viz",
        backend="altair",
    )

    loaded = load_history_dataframe(data_path, meta_path, expected_signature=signature)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_returns_empty_when_files_missing(tmp_path):
    """Test that missing files return empty dataframe."""
    data_path, meta_path = history_paths(tmp_path, "nonexistent")
    loaded = load_history_dataframe(data_path, meta_path, expected_signature="any")
    assert loaded.empty


def test_load_returns_empty_when_metadata_corrupted(tmp_path):
    """Test that corrupted metadata returns empty dataframe."""
    data_path, meta_path = history_paths(tmp_path, "corrupted")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text('{"step": 0, "value": 0.1}\n')
    meta_path.write_text("not valid json {{{")
    loaded = load_history_dataframe(data_path, meta_path, expected_signature="any")
    assert loaded.empty


def test_load_returns_empty_when_signature_mismatched(tmp_path):
    """Test that mismatched signature returns empty dataframe."""
    cfg = _simple_plot_config()
    signature = plot_config_signature(cfg)
    data_path, meta_path = history_paths(tmp_path, "mismatched")
    df = pd.DataFrame({"step": [0], "value": [0.1]})

    save_history_dataframe(
        df,
        data_path,
        meta_path,
        signature=signature,
        analysis="analysis",
        name="viz",
        backend="altair",
    )

    loaded = load_history_dataframe(data_path, meta_path, expected_signature="different_signature")
    assert loaded.empty


def test_load_returns_empty_when_data_corrupted(tmp_path):
    """Test that corrupted data file returns empty dataframe."""
    cfg = _simple_plot_config()
    signature = plot_config_signature(cfg)
    data_path, meta_path = history_paths(tmp_path, "data_corrupted")
    df = pd.DataFrame({"step": [0], "value": [0.1]})

    save_history_dataframe(
        df,
        data_path,
        meta_path,
        signature=signature,
        analysis="analysis",
        name="viz",
        backend="altair",
    )

    # Corrupt the data file
    data_path.write_text("not valid jsonl {{{")

    loaded = load_history_dataframe(data_path, meta_path, expected_signature=signature)
    assert loaded.empty


def test_plot_config_signature_handles_path_values():
    """Test that plot config signature can serialize Path objects."""
    layer = LayerConfig(
        geometry=GeometryConfig(type="line"),
        aesthetics=AestheticsConfig(
            x=ChannelAestheticsConfig(field="step", type="quantitative"),
            y=ChannelAestheticsConfig(field="value", type="quantitative"),
        ),
    )
    cfg = PlotConfig(
        backend="altair",
        data=DataConfig(source="main"),
        layers=[layer],
        size=PlotSizeConfig(width=400, height=200),
        guides=PlotLevelGuideConfig(),
    )
    # This should not raise even with complex nested objects
    sig = plot_config_signature(cfg)
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA256 hex length
