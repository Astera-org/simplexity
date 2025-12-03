"""Tests for the metrics module."""

from simplexity.metrics.metrics import RequiredFields, Requirements, combine_required_fields, combine_requirements


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
