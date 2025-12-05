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


def _induce_gradients(model: torch.nn.Module) -> None:
    """Induce gradients in the model."""
    for param in model.parameters():
        param.grad = torch.randn_like(param)


def test_init_all_metrics(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Test the MetricTracker class."""
    metric_tracker = MetricTracker(model=model, optimizer=optimizer)
    assert set(metric_tracker.metric_groups["all"]) == set(ALL_METRICS.keys())


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
    assert ALL_METRICS["learning_rate"].requirements.context_field_required("learning_rates")
    with patch("simplexity.metrics.metric_tracker.SIMPLEXITY_LOGGER.warning") as mock_warning:
        MetricTracker(metric_names=["learning_rate"])
        mock_warning.assert_called_once_with(
            "[Metrics] %s requires learning rates, but optimizer is not set in MetricTracker", "learning_rate"
        )

    assert ALL_METRICS["parameter_distance"].requirements.context_field_required("named_parameters")
    with patch("simplexity.metrics.metric_tracker.SIMPLEXITY_LOGGER.warning") as mock_warning:
        MetricTracker(metric_names=["parameter_distance"])
        mock_warning.assert_called_once_with(
            "[Metrics] %s requires gradients or named parameters, but model is not set in MetricTracker",
            "parameter_distance",
        )


def test_lazy_context_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Test the MetricTracker class."""
    metric_requirements = ALL_METRICS["learning_rate"].requirements
    metric_tracker = MetricTracker(metric_names=["learning_rate"], optimizer=optimizer)
    assert not metric_requirements.init_required
    assert metric_tracker.context.learning_rates == {}
    metric_tracker.step()
    assert not metric_requirements.step_required
    assert metric_tracker.context.learning_rates == {}
    metric_tracker.get_metrics()
    assert metric_requirements.compute_required
    assert metric_tracker.context.learning_rates != {}

    metric_requirements = ALL_METRICS["gradient_weighted_tokens"].requirements
    metric_tracker = MetricTracker(metric_names=["gradient_weighted_tokens"], model=model, optimizer=optimizer)
    assert not metric_requirements.init_required
    assert metric_tracker.context.gradients == {}
    _induce_gradients(model)
    metric_tracker.step()
    assert metric_requirements.step_required
    assert metric_tracker.context.gradients != {}

    metric_requirements = ALL_METRICS["parameter_distance"].requirements
    metric_tracker = MetricTracker(metric_names=["parameter_distance"], model=model)
    assert metric_requirements.init_required
    assert metric_tracker.context.named_parameters != {}
    metric_tracker.step()
    assert not metric_requirements.step_required
    assert metric_tracker.context.named_parameters == {}
    metric_tracker.get_metrics()
    assert metric_requirements.compute_required
    assert metric_tracker.context.named_parameters != {}


def test_cache_computed_metrics():
    """Test the MetricTracker class."""
    metric_tracker = MetricTracker(metric_names=["tokens"])
    with patch("simplexity.metrics.metrics.TokensMetric.compute") as mock_compute:
        num_steps = 3
        for _ in range(num_steps):
            metric_tracker.step(tokens=100)
            metric_tracker.get_metrics()
            metric_tracker.get_metrics()
        assert mock_compute.call_count == num_steps
