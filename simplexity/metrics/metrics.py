"""Metrics for tracking training progress."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any

import torch

from simplexity.utils.pytorch_utils import named_tensor_distance, tensor_stack_l2_norm

# pylint: disable=too-few-public-methods


@dataclass
class Context:
    """Immutable view of the information required by a metric for one step."""

    num_tokens: int = 0
    loss: float = float("inf")
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    gradients: Mapping[str, torch.Tensor] | None = None
    named_parameters: Mapping[str, torch.Tensor] | None = None


_RequiredFieldsBase = make_dataclass(
    "_RequiredFieldsBase",
    [(f.name, bool, field(default=False)) for f in fields(Context)],
    frozen=True,
    bases=(),
)


class RequiredFields(_RequiredFieldsBase):
    """Optional requirements for the context required by a metric.

    Fields automatically mirror those of Context, with each field being a bool
    indicating whether that context field is required.
    """

    @property
    def any_required(self) -> bool:
        """Return True if any of the required fields are required."""
        return any(getattr(self, field.name) for field in fields(self))


def combine_required_fields(required_fields_list: list[RequiredFields]) -> RequiredFields:
    """Combine multiple RequiredFields using OR logic.

    If any RequiredFields in the list requires a field, the combined result will require it.
    """
    if not required_fields_list:
        return RequiredFields()

    combined_dict = {
        field.name: any(getattr(required_field, field.name, False) for required_field in required_fields_list)
        for field in fields(RequiredFields)
    }

    return RequiredFields(**combined_dict)


@dataclass(frozen=True)
class Requirements:
    """Requirements for the context required by a metric."""

    init: RequiredFields = RequiredFields()
    step: RequiredFields = RequiredFields()
    compute: RequiredFields = RequiredFields()

    @property
    def init_required(self) -> bool:
        """Check if any of the required context fields are required for initialization."""
        return self.init.any_required

    @property
    def step_required(self) -> bool:
        """Check if any of the required context fields are required for stepping."""
        return self.step.any_required

    @property
    def compute_required(self) -> bool:
        """Check if any of the required context fields are required for computing."""
        return self.compute.any_required


def combine_requirements(requirements_list: list[Requirements]) -> Requirements:
    """Combine multiple Requirements using OR logic for each phase.

    For each phase (init, step, compute), combines the RequiredFields using OR logic.
    If any Requirements in the list requires a field in a phase, the combined result will require it.
    """
    if not requirements_list:
        return Requirements()

    combined_dict = {
        field.name: combine_required_fields([getattr(requirements, field.name) for requirements in requirements_list])
        for field in fields(Requirements)
    }
    return Requirements(**combined_dict)


class Metric:
    """Base class for metrics that provides default requirements attribute."""

    requirements: Requirements = Requirements()

    def __init__(self, _context: Context, **kwargs: Any) -> None:
        """Initialize the metric."""

    def step(self, _context: Context) -> None:
        """Step the metric state."""

    def compute(self, _context: Context) -> Mapping[str, float]:
        """Return the latest scalar(s)."""
        return {}  # pragma: no cover


class LearningRateMetric(Metric):
    """Reports learning rates for each optimizer param group."""

    requirements = Requirements(
        compute=RequiredFields(learning_rates=True),
    )

    def __init__(self, _context: Context, **_kwargs: Any) -> None:
        pass

    def step(self, _context: Context) -> None:
        """Step the learning rate metric."""

    def compute(self, context: Context) -> Mapping[str, float]:
        """Compute the learning rate metric."""
        values: MutableMapping[str, float] = {}
        if len(context.learning_rates) == 1:
            values["step/learning_rate"] = list(context.learning_rates.values())[0]
        else:
            for group_name, lr in context.learning_rates.items():
                values[f"learning_rate/{group_name}"] = lr
        return values


class TokensMetric(Metric):
    """Tracks instantaneous and cumulative token counts."""

    requirements = Requirements(
        step=RequiredFields(num_tokens=True),
        compute=RequiredFields(num_tokens=True),
    )

    def __init__(self, _context: Context, **_kwargs: Any) -> None:
        self.cumulative = 0.0

    def step(self, context: Context) -> None:
        """Step the token count metric."""
        self.cumulative += float(context.num_tokens)

    def compute(self, context: Context) -> Mapping[str, float]:
        """Compute the token count metric."""
        return {
            "step/tokens": context.num_tokens,
            "cum/tokens": self.cumulative,
        }


class LearningRateWeightedTokensMetric(Metric):
    """Tracks the learning rate weighted tokens."""

    requirements = Requirements(
        step=RequiredFields(num_tokens=True, learning_rates=True),
    )

    def __init__(self, _context: Context, **_kwargs: Any) -> None:
        self.weighted_tokens = 0.0
        self.cumulative = 0.0

    def step(self, context: Context) -> None:
        """Step the learning rate weighted tokens metric."""
        lr = list(context.learning_rates.values())[0]
        self.weighted_tokens = lr * float(context.num_tokens)
        self.cumulative += self.weighted_tokens

    def compute(self, _context: Context) -> Mapping[str, float]:
        """Compute the learning rate weighted tokens metric."""
        return {
            "step/lr_weighted_tokens": self.weighted_tokens,
            "cum/lr_weighted_tokens": self.cumulative,
        }


class GradientWeightedTokensMetric(Metric):
    """Tracks the gradient weighted tokens."""

    requirements = Requirements(
        step=RequiredFields(num_tokens=True, learning_rates=True, gradients=True),
    )

    def __init__(self, _context: Context, **_kwargs: Any) -> None:
        self.gradient_signal = 0.0
        self.cumulative_gradient_signal = 0.0
        self.fisher_proxy = 0.0
        self.cumulative_fisher_proxy = 0.0

    def step(self, context: Context) -> None:
        """Step the gradient weighted tokens metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        assert context.learning_rates is not None, "Learning rates are required for this metric"
        lr = list(context.learning_rates.values())[0]
        gradient_norm = tensor_stack_l2_norm(context.gradients.values())
        self.gradient_signal = lr * gradient_norm * float(context.num_tokens)
        self.cumulative_gradient_signal += self.gradient_signal
        self.fisher_proxy = gradient_norm**2 * float(context.num_tokens)
        self.cumulative_fisher_proxy += self.fisher_proxy

    def compute(self, _context: Context) -> Mapping[str, float]:
        """Compute the gradient weighted tokens metric."""
        return {
            "step/gradient_signal": self.gradient_signal,
            "cum/gradient_signal": self.cumulative_gradient_signal,
            "step/fisher_proxy": self.fisher_proxy,
            "cum/fisher_proxy": self.cumulative_fisher_proxy,
        }


class ParameterUpdateMetric(Metric):
    """Tracks the cumulative parameter update."""

    requirements = Requirements(
        init=RequiredFields(named_parameters=True),
        step=RequiredFields(named_parameters=True),
    )

    def __init__(self, context: Context, **_kwargs: Any) -> None:
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.previous_named_parameters: Mapping[str, torch.Tensor] = context.named_parameters
        self.step_norm = 0.0
        self.cumulative = 0.0

    def step(self, context: Context) -> None:
        """Step the cumulative parameter update metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.step_norm = named_tensor_distance(context.named_parameters, self.previous_named_parameters)
        self.cumulative += self.step_norm
        self.previous_named_parameters = context.named_parameters

    def compute(self, _context: Context) -> Mapping[str, float]:
        """Compute the update norm metric."""
        return {
            "step/param_update": self.step_norm,
            "cum/param_update": self.cumulative,
        }


class LossMetric(Metric):
    """Tracks the training loss."""

    requirements = Requirements(
        init=RequiredFields(loss=True),
        step=RequiredFields(loss=True),
        compute=RequiredFields(loss=True),
    )

    def __init__(self, context: Context, **kwargs: Any) -> None:
        self._step = 0
        self.initial_loss = context.loss
        self.optimal_loss = kwargs.get("optimal_loss", 0)
        self.min_loss = float("inf")
        self.ma_window_size = kwargs.get("ma_window_size", 100)
        self.ma_losses = [float("inf")] * self.ma_window_size
        self.ema_gamma = kwargs.get("ema_gamma", 0.9)
        self.ema_loss = float("inf")

    def step(self, context: Context) -> None:
        """Step the current loss metric."""
        if self.initial_loss == float("inf"):
            self.initial_loss = context.loss
        self.min_loss = min(self.min_loss, context.loss)
        self.ma_losses[self._step % self.ma_window_size] = context.loss
        if self.ema_loss == float("inf"):
            self.ema_loss = context.loss
        self.ema_loss = self.ema_gamma * self.ema_loss + (1 - self.ema_gamma) * context.loss
        self._step += 1

    def compute(self, context: Context) -> Mapping[str, float]:
        """Compute the current loss metric."""
        progress = (self.initial_loss - context.loss) / (self.initial_loss - self.optimal_loss)
        return {
            "loss/step": context.loss,
            "loss/min": self.min_loss,
            "loss/ma": sum(self.ma_losses) / self.ma_window_size,
            "loss/ema": self.ema_loss,
            "loss/progress_to_optimal": progress,
        }


class ParameterNormMetric(Metric):
    """Computes the global L2 norm over all parameters."""

    requirements = Requirements(
        compute=RequiredFields(named_parameters=True),
    )

    def __init__(self, _context: Context, **_kwargs: Any) -> None:
        pass

    def step(self, _context: Context) -> None:
        """Step the parameter norm metric."""

    def compute(self, context: Context) -> Mapping[str, float]:
        """Compute the parameter norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        norm = tensor_stack_l2_norm(context.named_parameters.values())
        weights = [tensor for name, tensor in context.named_parameters.items() if name.endswith("weight")]
        weight_norm = tensor_stack_l2_norm(weights)
        return {
            "model/params_norm": norm,
            "model/weights_norm": weight_norm,
        }


class ParameterDistanceMetric(Metric):
    """Reports the parameter space distance from the initial model state."""

    requirements = Requirements(
        init=RequiredFields(named_parameters=True),
        compute=RequiredFields(named_parameters=True),
    )

    def __init__(self, context: Context, **_kwargs: Any) -> None:
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.initial_named_parameters: Mapping[str, torch.Tensor] = context.named_parameters
        self.max_distance = 0.0

    def step(self, _context: Context) -> None:
        """Step the distance from initialization metric."""

    def compute(self, context: Context) -> Mapping[str, float]:
        """Compute the distance from initialization metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        distance = named_tensor_distance(context.named_parameters, self.initial_named_parameters)
        self.max_distance = max(self.max_distance, distance)
        return {
            "model/params_distance": distance,
            "model/max_params_distance": self.max_distance,
        }


ALL_METRICS: dict[str, type[Metric]] = {
    "learning_rate": LearningRateMetric,
    "tokens": TokensMetric,
    "learning_rate_weighted_tokens": LearningRateWeightedTokensMetric,
    "gradient_weighted_tokens": GradientWeightedTokensMetric,
    "parameter_update": ParameterUpdateMetric,
    "loss": LossMetric,
    "parameter_norm": ParameterNormMetric,
    "parameter_distance": ParameterDistanceMetric,
}
