"""Tests for the metrics module."""

from simplexity.metrics.metrics import (
    Context,
    LearningRateMetric,
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
    context = Context()
    metric = LearningRateMetric(context)
    metric.step(context)
    assert metric.compute(context) == {}

    context = Context(learning_rates={"lr": 0.01})
    assert metric.compute(context) == {"step/learning_rate": 0.01}

    context = Context(learning_rates={"lr1": 0.01, "lr2": 0.02})
    assert metric.compute(context) == {"learning_rate/lr1": 0.01, "learning_rate/lr2": 0.02}


def test_tokens():
    """Test the TokensMetric class."""
    context = Context()
    metric = TokensMetric(context)

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
