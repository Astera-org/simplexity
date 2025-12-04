"""Tests for the MetricTracker class."""

from unittest.mock import patch

import pytest
import torch

from simplexity.metrics.metric_tracker import MetricTracker
from simplexity.metrics.metrics import ALL_METRICS


@pytest.fixture
def model() -> torch.nn.Module:
    """Fixture for a test model."""
    return torch.nn.Linear(10, 10)


@pytest.fixture
def optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Fixture for a test optimizer."""
    return torch.optim.Adam(model.parameters(), lr=0.01)


def test_init_all_metrics(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Test the MetricTracker class."""
    metric_tracker = MetricTracker(model=model, optimizer=optimizer)
    assert set(metric_tracker.metric_groups["all"]) == set(ALL_METRICS.keys())
    assert metric_tracker.context.learning_rates == {}
    assert metric_tracker.context.gradients == {}
    assert metric_tracker.context.named_parameters == {}


def test_init_metric_list():
    """Test the MetricTracker class."""
    metric_tracker = MetricTracker(metric_names=["tokens", "loss"])
    assert set(metric_tracker.metric_groups["all"]) == {"tokens", "loss"}


def test_init_metric_dict(optimizer: torch.optim.Optimizer):
    """Test the MetricTracker class."""
    metric_tracker = MetricTracker(
        metric_names={"group_1": ["tokens", "loss"], "group_2": ["learning_rate", "learning_rate_weighted_tokens"]},
        optimizer=optimizer,
    )
    assert set(metric_tracker.metric_groups["group_1"]) == {"tokens", "loss"}
    assert set(metric_tracker.metric_groups["group_2"]) == {"learning_rate", "learning_rate_weighted_tokens"}
    assert set(metric_tracker.metric_groups["all"]) == {
        "tokens",
        "loss",
        "learning_rate",
        "learning_rate_weighted_tokens",
    }


def test_warn_missing_context():
    """Test the MetricTracker class."""
    with patch("simplexity.metrics.metric_tracker.SIMPLEXITY_LOGGER.warning") as mock_warning:
        MetricTracker(metric_names=["learning_rate"])
        mock_warning.assert_called_once_with(
            "[Metrics] %s requires learning rates, but optimizer is not set in MetricTracker", "learning_rate"
        )

    with patch("simplexity.metrics.metric_tracker.SIMPLEXITY_LOGGER.warning") as mock_warning:
        MetricTracker(metric_names=["parameter_distance"])
        mock_warning.assert_called_once_with(
            "[Metrics] %s requires gradients or named parameters, but model is not set in MetricTracker",
            "parameter_distance",
        )
