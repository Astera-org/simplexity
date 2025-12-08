"""Tests for the metrics module."""

import re
from unittest.mock import patch

import pytest
import torch

from simplexity.metrics.metrics import (
    ALL_METRICS,
    Context,
    GradientWeightedTokensMetric,
    LearningRateMetric,
    LearningRateWeightedTokensMetric,
    LossMetric,
    Metric,
    ParameterDistanceMetric,
    ParameterNormMetric,
    ParameterUpdateMetric,
    RequiredFields,
    Requirements,
    TokensMetric,
    combine_required_fields,
    combine_requirements,
    register_metric,
    unregister_metric,
)


class TestRequiredFields:
    """Tests for the RequiredFields class."""

    def test_required_fields(self):
        """Test the RequiredFields class."""
        required_fields = RequiredFields(learning_rates=True, named_parameters=False)
        assert required_fields.learning_rates
        assert not required_fields.gradients
        assert not required_fields.named_parameters

    def test_context_field_any_required(self):
        """Test the context_field_any_required method."""
        required_fields = RequiredFields()
        assert not required_fields.any_required
        required_fields = RequiredFields(learning_rates=True)
        assert required_fields.any_required

    def test_combine_required_fields(self):
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


class TestRequirements:
    """Tests for the Requirements class."""

    def test_requirements(self):
        """Test the Requirements class."""
        requirements = Requirements(step=RequiredFields(gradients=True))
        assert not requirements.init.gradients
        assert requirements.step.gradients
        assert not requirements.step.named_parameters

    def test_requirements_field_required(self):
        """Test the requirements_field_required method."""
        requirements = Requirements()
        assert not requirements.init_required
        assert not requirements.step_required
        assert not requirements.compute_required

        requirements = Requirements(init=RequiredFields(learning_rates=True))
        assert requirements.init_required
        assert not requirements.step_required

        requirements = Requirements(step=RequiredFields(gradients=True))
        assert requirements.step_required
        assert not requirements.compute_required

        requirements = Requirements(compute=RequiredFields(named_parameters=True))
        assert requirements.compute_required
        assert not requirements.init_required

    def test_requirements_context_field_required(self):
        """Test the requirements_context_field_required method."""
        requirements = Requirements(compute=RequiredFields(named_parameters=True))
        assert requirements.context_field_required("named_parameters")
        assert not requirements.context_field_required("gradients")

    def test_combine_requirements(self):
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


class TestMetrics:
    """Tests for the Metric classes."""

    def test_learning_rates(self):
        """Test the LearningRates class."""
        metric = LearningRateMetric(Context())
        metric.step(Context())
        assert metric.compute(Context()) == {}

        context = Context(learning_rates={"lr": 0.01})
        assert metric.compute(context) == {"step/learning_rate": 0.01}

        context = Context(learning_rates={"lr1": 0.01, "lr2": 0.02})
        assert metric.compute(context) == {"learning_rate/lr1": 0.01, "learning_rate/lr2": 0.02}

    def test_tokens(self):
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

    def test_learning_rate_weighted_tokens(self):
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

    def test_gradient_weighted_tokens(self):
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

    def test_parameter_update(self):
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

    def test_loss(self):
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

    def test_parameter_norm(self):
        """Test the ParameterNormMetric class."""
        metric = ParameterNormMetric(Context())
        metric.step(Context())

        context = Context(
            named_parameters={"param_1": torch.tensor([2.0, 4.0]), "param_2": torch.tensor([5.0, 6.0])}
        )  # norm is 9.0
        assert metric.compute(context) == {"model/params_norm": pytest.approx(9.0)}

    def test_parameter_distance(self):
        """Test the ParameterDistanceMetric class."""
        named_parameters = {
            "param_1": torch.tensor([1.4, 2.3]),
            "param_2": torch.tensor([3.2, 4.1]),
        }
        context = Context(named_parameters=named_parameters)
        metric = ParameterDistanceMetric(context)
        metric.step(Context())

        named_parameters = {
            "param_1": torch.tensor([3.4, 6.3]),  # delta is [2.0, 4.0]
            "param_2": torch.tensor([8.2, 10.1]),  # delta is [5.0, 6.0]
        }  # distance norm is sqrt(2.0**2 + 4.0**2 + 5.0**2 + 6.0**2) = 9.0
        context = Context(named_parameters=named_parameters)
        assert metric.compute(context) == {
            "model/params_distance": pytest.approx(9.0),
            "model/max_params_distance": pytest.approx(9.0),
        }

        named_parameters = {
            "param_1": torch.tensor([2.4, 3.3]),  # delta is [1.0, 1.0]
            "param_2": torch.tensor([4.2, 5.1]),  # delta is [1.0, 1.0]
        }  # distance norm is sqrt(1.0**2 + 1.0**2 + 1.0**2 + 1.0**2) = 2.0
        context = Context(named_parameters=named_parameters)
        assert metric.compute(context) == {
            "model/params_distance": pytest.approx(2.0),
            "model/max_params_distance": pytest.approx(9.0),
        }


class CustomMetric(Metric):
    """A custom metric for testing."""

    def compute(self, _context: Context) -> dict[str, float]:
        """Compute the custom metric."""
        return {"custom/value": 42.0}


class TestRegisterMetric:
    """Tests for the register_metric function."""

    def test_register_and_unregister(self):
        """Test registering a custom metric."""
        register_metric("test_custom", CustomMetric)

        assert "test_custom" in ALL_METRICS
        assert ALL_METRICS["test_custom"] is CustomMetric

        metric = ALL_METRICS["test_custom"](Context())
        assert metric.compute(Context()) == {"custom/value": 42.0}

        metric_class = unregister_metric("test_custom", ignore_missing=False)
        assert metric_class is CustomMetric
        assert "test_custom" not in ALL_METRICS

    def test_overwrite(self):
        """Test registering a metric with overwrite=True."""

        class CustomMetric1(Metric):
            """First custom metric."""

            def compute(self, _context: Context) -> dict[str, float]:
                return {"custom/value": 1.0}

        class CustomMetric2(Metric):
            """Second custom metric."""

            def compute(self, _context: Context) -> dict[str, float]:
                return {"custom/value": 2.0}

        register_metric("test_overwrite", CustomMetric1)
        assert "test_overwrite" in ALL_METRICS
        assert ALL_METRICS["test_overwrite"] is CustomMetric1
        metric = ALL_METRICS["test_overwrite"](Context())
        assert metric.compute(Context()) == {"custom/value": 1.0}

        with patch("simplexity.metrics.metrics.SIMPLEXITY_LOGGER.warning") as mock_warning:
            register_metric("test_overwrite", CustomMetric2, overwrite=True)
            mock_warning.assert_called_once_with(
                "[Metrics] '%s' of type '%s' is already registered. Overwriting it with type '%s'.",
                "test_overwrite",
                "CustomMetric1",
                "CustomMetric2",
            )

        assert "test_overwrite" in ALL_METRICS
        assert ALL_METRICS["test_overwrite"] is CustomMetric2
        metric = ALL_METRICS["test_overwrite"](Context())
        assert metric.compute(Context()) == {"custom/value": 2.0}

        metric_class = unregister_metric("test_overwrite", ignore_missing=False)
        assert metric_class is CustomMetric2
        assert "test_overwrite" not in ALL_METRICS

    def test_register_name_clash(self):
        """Test that registering a duplicate metric raises ValueError."""

        assert "loss" in ALL_METRICS
        with pytest.raises(
            ValueError, match=re.escape("[Metrics] 'loss' is already registered. Use overwrite=True to replace it.")
        ):
            register_metric("loss", CustomMetric)

    def test_register_metric_type_error_not_class(self):
        """Test that registering a non-class raises TypeError."""
        with pytest.raises(TypeError, match="metric_class must be a class \\(type\\)"):
            register_metric("test_not_class", "not a class")  # type: ignore[arg-type]

    def test_register_metric_type_error_not_subclass(self):
        """Test that registering a class that's not a Metric subclass raises TypeError."""

        class NotAMetric:  # pylint: disable=too-few-public-methods
            """A class that's not a Metric."""

        with pytest.raises(TypeError, match="metric_class must be a subclass of Metric"):
            register_metric("test_not_subclass", NotAMetric)  # type: ignore[arg-type]

    def test_unregister_metric_key_error(self):
        """Test that unregistering a non-existent metric raises KeyError."""
        with pytest.raises(
            KeyError,
            match=re.escape("[Metrics] 'nonexistent_metric' is not registered. Use ignore_missing=True to ignore."),
        ):
            unregister_metric("nonexistent_metric", ignore_missing=False)

    def test_unregister_metric_ignore_missing(self):
        """Test that unregistering a non-existent metric with ignore_missing=True does not raise an error."""
        with patch("simplexity.metrics.metrics.SIMPLEXITY_LOGGER.warning") as mock_warning:
            metric_class = unregister_metric("nonexistent_metric", ignore_missing=True)
            mock_warning.assert_called_once_with(
                "[Metrics] '%s' is not registered. Ignoring.",
                "nonexistent_metric",
            )
        assert metric_class is None
