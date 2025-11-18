"""Metrics for tracking training progress."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from simplexity.utils.pytorch_utils import named_tensor_distance, tensor_collection_l2_norm

# pylint: disable=too-few-public-methods


@dataclass
class MetricContext:
    """Immutable view of the information required by a metric for one step."""

    step: int
    num_tokens: int
    loss: float
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    gradients: Mapping[str, torch.Tensor] | None = None
    named_parameters: Mapping[str, torch.Tensor] | None = None


class Metric(Protocol):
    """Protocol for metrics that can be plugged into the tracker."""

    def __init__(self, **kwargs: Any) -> None: ...

    def update(self, context: MetricContext) -> None:
        """Update the metric state."""
        ...  # pylint: disable=unnecessary-ellipsis  # Protocol methods require ellipsis body

    def compute(self) -> Mapping[str, float]:
        """Return the latest scalar(s)."""
        ...  # pylint: disable=unnecessary-ellipsis  # Protocol methods require ellipsis body


def _tensor_collection_l2_norms(tensors: Iterable[Iterable[torch.Tensor]]) -> list[float]:
    """Compute an L2 norm across an iterable of iterables of tensors without stacking."""
    return [
        float(torch.linalg.vector_norm(torch.cat([t.detach().float().view(-1) for t in tensor]), ord=2))  # pylint: disable=not-callable
        for tensor in tensors
    ]


def _named_tensor_distances(tensors: Iterable[Mapping[str, torch.Tensor]]) -> Iterable[float]:
    """Compute an L2 distance between two named tensor collections."""
    return itertools.starmap(named_tensor_distance, itertools.pairwise(tensors))


class TokensMetric:
    """Tracks instantaneous and cumulative token counts."""

    update_every_step = True

    def __init__(self, **_kwargs: Any) -> None:
        self.num_tokens = 0.0
        self.cumulative = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the token count metric."""
        self.num_tokens = float(context.num_tokens)
        self.cumulative += float(context.num_tokens)

    def compute(self) -> Mapping[str, float]:
        """Compute the token count metric."""
        return {
            "tokens/raw": self.num_tokens,
            "tokens/raw/cumulative": self.cumulative,
        }


class LearningRateMetric:
    """Reports learning rates for each optimizer param group."""

    requires_learning_rates = True

    def __init__(self, **_kwargs: Any) -> None:
        self.learning_rates: Mapping[str, float] = {}

    def update(self, context: MetricContext) -> None:
        """Update the learning rate metric."""
        self.learning_rates = context.learning_rates

    def compute(self) -> Mapping[str, float]:
        """Compute the learning rate metric."""
        values: MutableMapping[str, float] = {}
        if len(self.learning_rates) == 1:
            values["lr"] = list(self.learning_rates.values())[0]
        else:
            for group_name, lr in self.learning_rates.items():
                values[f"lr/{group_name}"] = lr
        return values


class LearningRateWeightedTokensMetric:
    """Tracks the learning rate weighted tokens."""

    requires_learning_rates = True
    update_every_step = True

    def __init__(self, **_kwargs: Any) -> None:
        self.weighted_tokens = 0.0
        self.cumulative = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the learning rate weighted tokens metric."""
        lr = list(context.learning_rates.values())[0]
        self.weighted_tokens = lr * float(context.num_tokens)
        self.cumulative += self.weighted_tokens

    def compute(self) -> Mapping[str, float]:
        """Compute the learning rate weighted tokens metric."""
        return {
            "tokens/lr_weighted": self.weighted_tokens,
            "tokens/lr_weighted/cumulative": self.cumulative,
        }


class GradientWeightedTokensMetric:
    """Tracks the gradient weighted tokens."""

    requires_learning_rates = True
    requires_gradients = True
    update_every_step = True

    def __init__(self, **_kwargs: Any) -> None:
        self.gradients: list[Iterable[torch.Tensor]] = []
        self.lrs: list[float] = []
        self.num_tokens: list[float] = []
        self.cumulative = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the gradient weighted tokens metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        self.gradients.append(context.gradients.values())
        self.lrs.append(list(context.learning_rates.values())[0])
        self.num_tokens.append(float(context.num_tokens))

    def compute(self) -> Mapping[str, float]:
        """Compute the gradient weighted tokens metric."""
        gradient_norms = _tensor_collection_l2_norms(self.gradients)
        weighted_tokens = sum(
            lr * gradient_norm * num_tokens
            for lr, gradient_norm, num_tokens in zip(self.lrs, gradient_norms, self.num_tokens, strict=True)
        )
        self.cumulative += weighted_tokens
        self.gradients.clear()
        self.lrs.clear()
        self.num_tokens.clear()
        return {
            "tokens/gradient_weighted": weighted_tokens,
            "tokens/gradient_weighted/cumulative": self.cumulative,
        }


class CurrentLossMetric:
    """Logs the instantaneous training loss."""

    update_every_step = True

    def __init__(self, **kwargs: Any) -> None:
        self.loss = float("inf")
        self.min_loss = float("inf")
        self.ma_window_size = kwargs.get("ma_window_size", 100)
        self.ma_losses = [float("inf")] * self.ma_window_size
        self.ema_gamma = kwargs.get("ema_gamma", 0.9)
        self.ema_loss = float("inf")

    def update(self, context: MetricContext) -> None:
        """Update the current loss metric."""
        self.loss = context.loss
        self.min_loss = min(self.min_loss, context.loss)
        self.ma_losses[context.step % self.ma_window_size] = context.loss
        if self.ema_loss == float("inf"):
            self.ema_loss = context.loss
        self.ema_loss = self.ema_gamma * self.ema_loss + (1 - self.ema_gamma) * context.loss

    def compute(self) -> Mapping[str, float]:
        """Compute the current loss metric."""
        return {
            "loss": self.loss,
            "loss/min": self.min_loss,
            "loss/ma": sum(self.ma_losses) / self.ma_window_size,
            "loss/ema": self.ema_loss,
        }


class ParameterNormMetric:
    """Computes the global L2 norm over all parameters."""

    requires_named_parameters = True

    def __init__(self, **_kwargs: Any) -> None:
        self.named_parameters: Mapping[str, torch.Tensor] = {}

    def update(self, context: MetricContext) -> None:
        """Update the parameter norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.named_parameters = context.named_parameters

    def compute(self) -> Mapping[str, float]:
        """Compute the parameter norm metric."""
        norm = tensor_collection_l2_norm(self.named_parameters.values())
        return {"params/l2_norm": norm}


class WeightNormMetric:
    """Computes the L2 norm over parameters whose name ends with 'weight'."""

    requires_named_parameters = True

    def __init__(self, **_kwargs: Any) -> None:
        self.named_parameters: Mapping[str, torch.Tensor] = {}

    def update(self, context: MetricContext) -> None:
        """Update the weight norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.named_parameters = context.named_parameters

    def compute(self) -> Mapping[str, float]:
        """Compute the weight norm metric."""
        weight_tensors = [tensor for name, tensor in self.named_parameters.items() if name.endswith("weight")]
        norm = tensor_collection_l2_norm(weight_tensors)
        return {"params/weights_l2_norm": norm}


class DistanceFromInitializationMetric:
    """Reports the parameter space distance from the initial model state."""

    requires_named_parameters = True

    def __init__(self, **kwargs: Any) -> None:
        initial_named_parameters = kwargs.get("named_parameters")
        assert initial_named_parameters is not None, "Named parameters are required for this metric"
        self.initial_named_parameters: Mapping[str, torch.Tensor] = initial_named_parameters
        self.named_parameters: Mapping[str, torch.Tensor] = {}
        self.max_distance = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the distance from initialization metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.named_parameters = context.named_parameters

    def compute(self) -> Mapping[str, float]:
        """Compute the distance from initialization metric."""
        distance = named_tensor_distance(self.named_parameters, self.initial_named_parameters)
        self.max_distance = max(self.max_distance, distance)
        return {
            "params/distance_from_init": distance,
            "params/distance_from_init/max": self.max_distance,
        }


class CumulativeParameterUpdateMetric:
    """Tracks the cumulative parameter update."""

    requires_named_parameters = True
    update_every_step = True

    def __init__(self, **kwargs: Any) -> None:
        named_parameters = kwargs.get("named_parameters")
        assert named_parameters is not None, "Named parameters are required for this metric"
        self.named_parameters: list[Mapping[str, torch.Tensor]] = [named_parameters]
        self.cumulative = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the cumulative parameter update metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        self.named_parameters.append(context.named_parameters)

    def compute(self) -> Mapping[str, float]:
        """Compute the update norm metric."""
        step_norm = sum(_named_tensor_distances(self.named_parameters))
        self.cumulative += step_norm
        self.named_parameters = self.named_parameters[-1:]
        return {
            "params/update_l2_norm": step_norm,
            "params/update_l2_norm/cumulative": self.cumulative,
        }


class FisherInformationMetric:
    """Tracks the Fisher information."""

    requires_gradients = True
    update_every_step = True

    def __init__(self, **_kwargs: Any) -> None:
        self.gradients: list[Iterable[torch.Tensor]] = []
        self.cumulative = 0.0

    def update(self, context: MetricContext) -> None:
        """Update the Fisher information metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        self.gradients.append(context.gradients.values())

    def compute(self) -> Mapping[str, float]:
        """Compute the Fisher information metric."""
        gradient_norms = _tensor_collection_l2_norms(self.gradients)
        fisher_information = sum(gradient_norm**2 for gradient_norm in gradient_norms)
        self.cumulative += fisher_information
        self.gradients.clear()
        return {
            "params/fisher_information": fisher_information,
            "params/fisher_information/cumulative": self.cumulative,
        }


class LossProgressMetric:
    """Tracks the progress towards the optimal loss."""

    def __init__(self, **kwargs: Any) -> None:
        self.initial_loss = kwargs.get("initial_loss", float("inf"))
        self.optimal_loss = kwargs.get("optimal_loss", 0)
        self.current_loss = float("inf")

    def update(self, context: MetricContext) -> None:
        """Update the loss progress metric."""
        if self.initial_loss == float("inf"):
            self.initial_loss = context.loss
        self.current_loss = context.loss

    def compute(self) -> Mapping[str, float]:
        """Compute the loss progress metric."""
        progress = (self.initial_loss - self.current_loss) / (self.initial_loss - self.optimal_loss)
        return {"loss/progress_to_optimal": progress}


ALL_METRICS = {
    "tokens": TokensMetric,
    "lr": LearningRateMetric,
    "learning_rate_weighted_tokens": LearningRateWeightedTokensMetric,
    "gradient_weighted_tokens": GradientWeightedTokensMetric,
    "loss": CurrentLossMetric,
    "parameter_norm": ParameterNormMetric,
    "weight_norm": WeightNormMetric,
    "distance_from_initialization": DistanceFromInitializationMetric,
    "cumulative_parameter_update": CumulativeParameterUpdateMetric,
    "fisher_information": FisherInformationMetric,
    "loss_progress_to_optimal": LossProgressMetric,
}
