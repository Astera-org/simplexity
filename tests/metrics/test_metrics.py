"""Tests for the metrics module."""

import pytest
import torch

from simplexity.metrics.metrics import (
    Context,
    GradientWeightedTokensMetric,
    LearningRateMetric,
    LearningRateWeightedTokensMetric,
    LossMetric,
    ParameterUpdateMetric,
    RequiredFields,
    Requirements,
    TokensMetric,
    combine_required_fields,
    combine_requirements,
)


def test_required_fields():
    """Test the RequiredFields class."""
    required_fields = RequiredFields(learning_rates=True, named_parameters=False)
    assert required_fields.learning_rates
    assert not required_fields.gradients
    assert not required_fields.named_parameters


def test_combine_required_fields():
    """Test the combine_required_fields function."""
    required_fields_list = [
        RequiredFields(learning_rates=True),
        RequiredFields(learning_rates=False, gradients=True, named_parameters=False),
        RequiredFields(),
    ]
    combined_required_fields = combine_required_fields(required_fields_list)
    assert combined_required_fields.learning_rates
    assert combined_required_fields.gradients
    assert not combined_required_fields.named_parameters


def test_requirements():
    """Test the Requirements class."""
    requirements = Requirements(step=RequiredFields(gradients=True))
    assert not requirements.init.gradients
    assert requirements.step.gradients
    assert not requirements.step.named_parameters


def test_combine_requirements():
    """Test the combine_requirements function."""
    requirements_list = [
        Requirements(init=RequiredFields(learning_rates=True)),
        Requirements(step=RequiredFields(gradients=True)),
        Requirements(compute=RequiredFields(named_parameters=True)),
    ]
    combined_requirements = combine_requirements(requirements_list)
    assert combined_requirements.init.learning_rates
    assert combined_requirements.step.gradients
    assert combined_requirements.compute.named_parameters


def test_learning_rates():
    """Test the LearningRates class."""
    metric = LearningRateMetric(Context())
    metric.step(Context())
    assert metric.compute(Context()) == {}

    context = Context(learning_rates={"lr": 0.01})
    assert metric.compute(context) == {"step/learning_rate": 0.01}

    context = Context(learning_rates={"lr1": 0.01, "lr2": 0.02})
    assert metric.compute(context) == {"learning_rate/lr1": 0.01, "learning_rate/lr2": 0.02}


def test_tokens():
    """Test the TokensMetric class."""
    metric = TokensMetric(Context())

    context = Context(num_tokens=20)
    metric.step(context)
    computed = metric.compute(context)
    assert computed["step/tokens"] == 20
    assert computed["cum/tokens"] == 20
    assert computed["step/tokens_per_second"] > 0
    assert computed["cum/tokens_per_second"] > 0

    context = Context(num_tokens=30)
    metric.step(context)
    computed = metric.compute(context)
    assert computed["step/tokens"] == 30
    assert computed["cum/tokens"] == 50
    assert computed["step/tokens_per_second"] > 0
    assert computed["cum/tokens_per_second"] > 0


def test_learning_rate_weighted_tokens():
    """Test the LearningRateWeightedTokensMetric class."""
    context = Context()
    metric = LearningRateWeightedTokensMetric(context)

    context = Context(num_tokens=20, learning_rates={"lr": 0.01})
    metric.step(context)
    assert metric.compute(context) == {
        "step/lr_weighted_tokens": 0.2,
        "cum/lr_weighted_tokens": 0.2,
    }

    context = Context(num_tokens=30, learning_rates={"lr": 0.02})
    metric.step(context)
    assert metric.compute(context) == {
        "step/lr_weighted_tokens": 0.6,
        "cum/lr_weighted_tokens": 0.8,
    }


def test_gradient_weighted_tokens():
    """Test the GradientWeightedTokensMetric class."""
    context = Context()
    metric = GradientWeightedTokensMetric(context)

    learning_rates = {"lr": 0.01}
    gradients = {"grad_1": torch.tensor([2.0, 4.0]), "grad_2": torch.tensor([5.0, 6.0])}  # gradient norm is 9.0
    context = Context(num_tokens=20, learning_rates=learning_rates, gradients=gradients)
    metric.step(context)

    assert metric.compute(Context()) == {
        "step/gradient_signal": pytest.approx(1.8),  # 0.01 * 9.0 * 20
        "cum/gradient_signal": pytest.approx(1.8),
        "step/fisher_proxy": pytest.approx(1620.0),  # 9.0**2 * 20
        "cum/fisher_proxy": pytest.approx(1620.0),
    }

    context = Context(num_tokens=30, learning_rates={"lr": 0.02}, gradients={"grad": torch.tensor([1.5])})
    metric.step(context)
    assert metric.compute(context) == {
        "step/gradient_signal": pytest.approx(0.9),  # 0.02 * 1.5 * 30
        "cum/gradient_signal": pytest.approx(2.7),  # 1.8 + 0.9
        "step/fisher_proxy": pytest.approx(67.5),  # 1.5**2 * 30
        "cum/fisher_proxy": pytest.approx(1687.5),  # 1620 + 67.5
    }


def test_parameter_update():
    """Test the ParameterUpdateMetric class."""
    context = Context(
        named_parameters={
            "param_1": torch.tensor([4.1, 3.2]),
            "param_2": torch.tensor([2.3, 1.4]),
        }
    )
    metric = ParameterUpdateMetric(context)

    context = Context(
        named_parameters={
            "param_1": torch.tensor([6.1, 7.2]),  # delta is [2.0, 4.0]
            "param_2": torch.tensor([7.3, 7.4]),  # delta is [5.0, 6.0]
        }
    )  # distance norm is sqrt(2.0**2 + 4.0**2 + 5.0**2 + 6.0**2) = 9.0
    metric.step(context)

    assert metric.compute(Context()) == {
        "step/param_update": pytest.approx(9.0),
        "cum/param_update": pytest.approx(9.0),
    }

    context = Context(
        named_parameters={
            "param_1": torch.tensor([7.1, 8.2]),  # delta is [1.0, 1.0]
            "param_2": torch.tensor([8.3, 8.4]),  # delta is [1.0, 1.0]
        }
    )  # distance norm is sqrt(1.0**2 + 1.0**2 + 1.0**2 + 1.0**2) = 2.0
    metric.step(context)
    assert metric.compute(context) == {
        "step/param_update": pytest.approx(2.0),
        "cum/param_update": pytest.approx(11.0),
    }


def test_loss():
    """Test the LossMetric class."""
    context = Context(loss=2.0)
    metric = LossMetric(context, optimal_loss=1.0, ma_window_size=2, ema_gamma=0.9)

    context = Context(loss=1.4)
    metric.step(context)
    assert metric.compute(context) == {
        "loss/step": pytest.approx(1.4),
        "loss/min": pytest.approx(1.4),
        "loss/ma": float("inf"),
        "loss/ema": pytest.approx(1.4),
        "loss/progress_to_optimal": pytest.approx(0.4),
    }

    context = Context(loss=1.7)
    metric.step(context)
    assert metric.compute(context) == {
        "loss/step": pytest.approx(1.7),
        "loss/min": pytest.approx(1.4),
        "loss/ma": pytest.approx(1.55),  # (1.4 + 1.7) / 2
        "loss/ema": pytest.approx(1.43),  # 0.9 * 1.4 + 0.1 * 1.7
        "loss/progress_to_optimal": pytest.approx(0.7),
    }

    context = Context(loss=1.1)
    metric.step(context)
    assert metric.compute(context) == {
        "loss/step": pytest.approx(1.1),
        "loss/min": pytest.approx(1.1),
        "loss/ma": pytest.approx(1.4),  # (1.7 + 1.1) / 2
        "loss/ema": pytest.approx(1.397),  # 0.9 * 1.43 + 0.1 * 1.1
        "loss/progress_to_optimal": pytest.approx(0.1),
    }
