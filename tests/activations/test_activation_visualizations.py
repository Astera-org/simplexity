"""Tests for activation visualization functions."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# pylint: enable=all

import numpy as np
import pytest

from simplexity.activations.activation_visualizations import (
    PreparedMetadata,
    _compute_aggregation,
    _parse_scalar_expression,
    _render_title_template,
    build_visualization_payloads,
)
from simplexity.activations.visualization_configs import build_activation_visualization_config
from simplexity.exceptions import ConfigValidationError


class TestParseScalarExpression:
    """Tests for _parse_scalar_expression function."""

    def test_simple_key(self):
        """Test parsing a simple scalar key without aggregation."""
        key, agg = _parse_scalar_expression("my_scalar")
        assert key == "my_scalar"
        assert agg is None

    def test_min_aggregation(self):
        """Test parsing min aggregation."""
        key, agg = _parse_scalar_expression("min(my_scalar)")
        assert key == "my_scalar"
        assert agg == "min"

    def test_max_aggregation(self):
        """Test parsing max aggregation."""
        key, agg = _parse_scalar_expression("max(my_scalar)")
        assert key == "my_scalar"
        assert agg == "max"

    def test_avg_aggregation(self):
        """Test parsing avg aggregation."""
        key, agg = _parse_scalar_expression("avg(my_scalar)")
        assert key == "my_scalar"
        assert agg == "avg"

    def test_mean_aggregation(self):
        """Test parsing mean aggregation."""
        key, agg = _parse_scalar_expression("mean(my_scalar)")
        assert key == "my_scalar"
        assert agg == "mean"

    def test_latest_aggregation(self):
        """Test parsing latest aggregation."""
        key, agg = _parse_scalar_expression("latest(my_scalar)")
        assert key == "my_scalar"
        assert agg == "latest"

    def test_first_aggregation(self):
        """Test parsing first aggregation."""
        key, agg = _parse_scalar_expression("first(my_scalar)")
        assert key == "my_scalar"
        assert agg == "first"

    def test_last_aggregation(self):
        """Test parsing last aggregation."""
        key, agg = _parse_scalar_expression("last(my_scalar)")
        assert key == "my_scalar"
        assert agg == "last"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        key, agg = _parse_scalar_expression("  min( my_scalar )  ")
        assert key == "my_scalar"
        assert agg == "min"


class TestComputeAggregation:
    """Tests for _compute_aggregation function."""

    def test_min(self):
        """Test min aggregation."""
        history = [(0, 5.0), (1, 3.0), (2, 7.0)]
        result = _compute_aggregation(history, "min")
        assert result == 3.0

    def test_max(self):
        """Test max aggregation."""
        history = [(0, 5.0), (1, 3.0), (2, 7.0)]
        result = _compute_aggregation(history, "max")
        assert result == 7.0

    def test_avg(self):
        """Test avg aggregation."""
        history = [(0, 3.0), (1, 6.0), (2, 9.0)]
        result = _compute_aggregation(history, "avg")
        assert result == 6.0

    def test_mean(self):
        """Test mean aggregation (alias for avg)."""
        history = [(0, 3.0), (1, 6.0), (2, 9.0)]
        result = _compute_aggregation(history, "mean")
        assert result == 6.0

    def test_latest(self):
        """Test latest aggregation."""
        history = [(0, 5.0), (1, 3.0), (2, 7.0)]
        result = _compute_aggregation(history, "latest")
        assert result == 7.0

    def test_last(self):
        """Test last aggregation (alias for latest)."""
        history = [(0, 5.0), (1, 3.0), (2, 7.0)]
        result = _compute_aggregation(history, "last")
        assert result == 7.0

    def test_first(self):
        """Test first aggregation."""
        history = [(0, 5.0), (1, 3.0), (2, 7.0)]
        result = _compute_aggregation(history, "first")
        assert result == 5.0

    def test_empty_history_raises(self):
        """Test that empty history raises error."""
        with pytest.raises(ConfigValidationError, match="empty history"):
            _compute_aggregation([], "min")

    def test_unknown_function_raises(self):
        """Test that unknown function raises error."""
        history = [(0, 5.0)]
        with pytest.raises(ConfigValidationError, match="Unknown aggregation"):
            _compute_aggregation(history, "unknown_func")


class TestRenderTitleTemplate:
    """Tests for _render_title_template function."""

    def test_none_title_returns_none(self):
        """Test that None title returns None."""
        result = _render_title_template(None, None, {}, {})
        assert result is None

    def test_simple_title_no_substitution(self):
        """Test title without placeholders."""
        result = _render_title_template("My Title", None, {}, {})
        assert result == "My Title"

    def test_title_with_scalar_substitution(self):
        """Test title with scalar value substitution."""
        title = "Loss: {loss:.3f}"
        title_scalars = {"loss": "test/loss"}
        scalars = {"test/loss": 0.12345}
        result = _render_title_template(title, title_scalars, scalars, {})
        assert result == "Loss: 0.123"

    def test_title_with_history_aggregation(self):
        """Test title with scalar history aggregation."""
        title = "Min Loss: {min_loss:.2f}"
        title_scalars = {"min_loss": "min(test/loss)"}
        scalars = {}
        scalar_history = {"test/loss": [(0, 0.5), (1, 0.3), (2, 0.4)]}
        result = _render_title_template(title, title_scalars, scalars, scalar_history)
        assert result == "Min Loss: 0.30"


class TestBuildVisualizationPayloads:
    """Tests for build_visualization_payloads function."""

    @pytest.fixture
    def basic_metadata(self):
        """Create basic metadata for testing."""
        return PreparedMetadata(
            sequences=[(1, 2), (1, 3)],
            steps=np.array([2, 2]),
            select_last_token=False,
        )

    @pytest.fixture
    def basic_viz_config(self):
        """Create a basic visualization config."""
        return build_activation_visualization_config(
            {
                "name": "test_viz",
                "data_mapping": {
                    "mappings": {
                        "pc0": {"source": "projections", "key": "pca", "component": 0},
                    },
                },
                "layer": {
                    "geometry": {"type": "point"},
                    "aesthetics": {
                        "x": {"field": "pc0", "type": "quantitative"},
                    },
                },
            }
        )

    def test_builds_payload_with_projections(self, basic_metadata, basic_viz_config):
        """Test building a payload with projection data."""
        projections = {"layer_0_pca": np.array([[1.0, 2.0], [3.0, 4.0]])}
        payloads = build_visualization_payloads(
            analysis_name="test",
            viz_cfgs=[basic_viz_config],
            default_backend="altair",
            prepared_metadata=basic_metadata,
            weights=np.array([0.5, 0.5]),
            belief_states=None,
            projections=projections,
            scalars={},
            scalar_history={},
            scalar_history_step=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload.name == "test_viz"
        assert not payload.dataframe.empty

    def test_builds_payload_with_belief_states(self, basic_metadata):
        """Test building a payload with belief state data."""
        viz_config = build_activation_visualization_config(
            {
                "name": "belief_viz",
                "data_mapping": {
                    "mappings": {
                        "belief_0": {"source": "belief_states", "component": 0},
                    },
                },
                "layer": {
                    "geometry": {"type": "point"},
                    "aesthetics": {
                        "x": {"field": "belief_0", "type": "quantitative"},
                    },
                },
            }
        )
        belief_states = np.array([[0.5, 0.5], [0.3, 0.7]])
        payloads = build_visualization_payloads(
            analysis_name="test",
            viz_cfgs=[viz_config],
            default_backend="altair",
            prepared_metadata=basic_metadata,
            weights=np.array([0.5, 0.5]),
            belief_states=belief_states,
            projections={},
            scalars={},
            scalar_history={},
            scalar_history_step=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )
        assert len(payloads) == 1
        assert "belief_0" in payloads[0].dataframe.columns

    def test_handles_multiple_configs(self, basic_metadata):
        """Test building multiple payloads from multiple configs."""
        configs = [
            build_activation_visualization_config(
                {
                    "name": "viz_1",
                    "data_mapping": {"mappings": {"pc0": {"source": "projections", "key": "pca", "component": 0}}},
                    "layer": {
                        "geometry": {"type": "point"},
                        "aesthetics": {"x": {"field": "pc0", "type": "quantitative"}},
                    },
                }
            ),
            build_activation_visualization_config(
                {
                    "name": "viz_2",
                    "data_mapping": {"mappings": {"pc1": {"source": "projections", "key": "pca", "component": 1}}},
                    "layer": {
                        "geometry": {"type": "point"},
                        "aesthetics": {"x": {"field": "pc1", "type": "quantitative"}},
                    },
                }
            ),
        ]
        projections = {"layer_0_pca": np.array([[1.0, 2.0], [3.0, 4.0]])}
        payloads = build_visualization_payloads(
            analysis_name="test",
            viz_cfgs=configs,
            default_backend="altair",
            prepared_metadata=basic_metadata,
            weights=np.array([0.5, 0.5]),
            belief_states=None,
            projections=projections,
            scalars={},
            scalar_history={},
            scalar_history_step=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )
        assert len(payloads) == 2
        assert payloads[0].name == "viz_1"
        assert payloads[1].name == "viz_2"
