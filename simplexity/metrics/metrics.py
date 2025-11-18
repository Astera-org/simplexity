"""Metrics for tracking training progress."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

# pylint: disable=too-few-public-methods


@dataclass
class MetricContext:
    """Immutable view of the information required by a metric for one step."""

    step: int
    batch_tokens: int
    total_tokens: int
    loss: float
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    gradients: Mapping[str, torch.Tensor] | None = None
    named_parameters: Mapping[str, torch.Tensor] | None = None


class TrainingMetric(Protocol):
    """Protocol for metrics that can be plugged into the tracker."""

    def __init__(self, **kwargs: Any) -> None: ...

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Update the metric state and return the latest scalar(s)."""
        ...  # pylint: disable=unnecessary-ellipsis  # Protocol methods require ellipsis body


def _tensor_collection_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    """Compute an L2 norm across an iterable of tensors without stacking.

    We detach and cast to float to avoid autograd interactions and dtype
    mismatches, skip empty tensors, and accumulate sums on CPU. This mirrors
    concatenating the tensors and calling ``torch.linalg.norm`` but without the
    intermediate allocation, which keeps metric computations lightweight even
    when parameters/gradients live on different devices or have differing
    shapes.
    """
    total = 0.0
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        total += float(torch.sum(torch.square(tensor.detach().float())))
    return total**0.5


def _named_tensor_distance(current: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float:
    """Compute an L2 distance between two named tensor collections.

    Parameters are compared by name without stacking tensors into a single
    structure, which keeps the computation memory-efficient. Tensors are
    detached, moved to CPU, and cast to float so that distance metrics stay off
    the autograd graph and remain robust even if current and reference tensors
    live on different devices or use different dtypes. Using built-in norms
    would require materializing aligned tensors (or concatenations) per
    parameter pair on the same device/dtype, which is both more memory hungry
    and brittle when models evolve.
    """
    total = 0.0
    for name, tensor in current.items():
        ref_tensor = reference.get(name)
        if ref_tensor is None or tensor.numel() == 0:
            continue
        curr_cpu = tensor.detach().float().cpu()
        diff = curr_cpu - ref_tensor.float()
        total += float(torch.sum(torch.square(diff)))
    return total**0.5


class TokensMetric:
    """Tracks instantaneous and cumulative token counts."""

    def __init__(self, **_kwargs: Any) -> None: ...

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the token count metric."""
        return {
            "tokens/batch": float(context.batch_tokens),
            "tokens/total": float(context.total_tokens),
        }


class LearningRateMetric:
    """Reports learning rates for each optimizer param group."""

    requires_learning_rates = True

    def __init__(self, **_kwargs: Any) -> None: ...

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the learning rate metric."""
        values: MutableMapping[str, float] = {}
        if len(context.learning_rates) == 1:
            values["lr"] = list(context.learning_rates.values())[0]
        else:
            for group_name, lr in context.learning_rates.items():
                values[f"lr/{group_name}"] = lr
        return values


class LearningRateWeightedTokensMetric:
    """Tracks the learning rate weighted tokens."""

    requires_learning_rates = True

    def __init__(self, **_kwargs: Any) -> None:
        self.cumulative = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the learning rate weighted tokens metric."""
        lr = list(context.learning_rates.values())[0]
        weighted_tokens = lr * float(context.batch_tokens)
        self.cumulative += weighted_tokens
        return {
            "tokens/lr_weighted": weighted_tokens,
            "tokens/lr_weighted/cumulative": self.cumulative,
        }


class GradientWeightedTokensMetric:
    """Tracks the gradient weighted tokens."""

    requires_learning_rates = True
    requires_gradients = True

    def __init__(self, **_kwargs: Any) -> None:
        self.cumulative = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the gradient weighted tokens metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        gradient_norm = _tensor_collection_l2_norm(context.gradients.values())
        lr = list(context.learning_rates.values())[0]
        weighted_tokens = lr * gradient_norm * float(context.batch_tokens)
        self.cumulative += weighted_tokens
        return {
            "tokens/gradient_weighted": weighted_tokens,
            "tokens/gradient_weighted/cumulative": self.cumulative,
        }


class CurrentLossMetric:
    """Logs the instantaneous training loss."""

    def __init__(self, **kwargs: Any) -> None:
        self.min_loss = float("inf")
        self.ma_window_size = kwargs.get("ma_window_size", 100)
        self.ma_losses = [float("inf")] * self.ma_window_size
        self.ema_gamma = kwargs.get("ema_gamma", 0.9)
        self.ema_loss = float("inf")

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the current loss metric."""
        self.min_loss = min(self.min_loss, context.loss)
        self.ma_losses[context.step % self.ma_window_size] = context.loss
        if self.ema_loss == float("inf"):
            self.ema_loss = context.loss
        self.ema_loss = self.ema_gamma * self.ema_loss + (1 - self.ema_gamma) * context.loss
        return {
            "loss": context.loss,
            "loss/min": self.min_loss,
            "loss/ma": sum(self.ma_losses) / self.ma_window_size,
            "loss/ema": self.ema_loss,
        }


class ParameterNormMetric:
    """Computes the global L2 norm over all parameters."""

    requires_named_parameters = True

    def __init__(self, **_kwargs: Any) -> None: ...

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the parameter norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        return {"params/l2_norm": _tensor_collection_l2_norm(context.named_parameters.values())}


class WeightNormMetric:
    """Computes the L2 norm over parameters whose name ends with 'weight'."""

    requires_named_parameters = True

    def __init__(self, **_kwargs: Any) -> None: ...

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the weight norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        weight_tensors = [tensor for name, tensor in context.named_parameters.items() if name.endswith("weight")]
        return {"params/weights_l2_norm": _tensor_collection_l2_norm(weight_tensors)}


class DistanceFromInitializationMetric:
    """Reports the parameter space distance from the initial model state."""

    requires_named_parameters = True

    def __init__(self, **kwargs: Any) -> None:
        initial_named_parameters = kwargs.get("named_parameters")
        assert initial_named_parameters is not None, "Named parameters are required for this metric"
        self.initial_named_parameters: Mapping[str, torch.Tensor] = initial_named_parameters
        self.max_distance = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the distance from initialization metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        distance = _named_tensor_distance(context.named_parameters, self.initial_named_parameters)
        self.max_distance = max(self.max_distance, distance)
        return {
            "params/distance_from_init": distance,
            "params/distance_from_init/max": self.max_distance,
        }


class CumulativeParameterUpdateMetric:
    """Tracks the cumulative parameter update."""

    requires_named_parameters = True

    def __init__(self, **kwargs: Any) -> None:
        named_parameters = kwargs.get("named_parameters")
        assert named_parameters is not None, "Named parameters are required for this metric"
        self.previous_named_parameters: Mapping[str, torch.Tensor] = named_parameters
        self.cumulative = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the update norm metric."""
        assert context.named_parameters is not None, "Named parameters are required for this metric"
        step_norm = _named_tensor_distance(context.named_parameters, self.previous_named_parameters)
        self.cumulative += step_norm
        self.previous_named_parameters = context.named_parameters
        return {
            "params/update_l2_norm": step_norm,
            "params/update_l2_norm/cumulative": self.cumulative,
        }


class FisherInformationMetric:
    """Tracks the Fisher information."""

    requires_gradients = True

    def __init__(self, **_kwargs: Any) -> None:
        self.cumulative = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the Fisher information metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        gradient_norm = _tensor_collection_l2_norm(context.gradients.values())
        fisher_information = gradient_norm**2
        self.cumulative += fisher_information
        return {
            "params/fisher_information": fisher_information,
            "params/fisher_information/cumulative": self.cumulative,
        }


class LossProgressMetric:
    """Tracks the progress towards the optimal loss."""

    def __init__(self, **kwargs: Any) -> None:
        self.initial_loss = kwargs.get("initial_loss", float("inf"))
        self.optimal_loss = kwargs.get("optimal_loss", 0)

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the loss progress metric."""
        if self.initial_loss == float("inf"):
            self.initial_loss = context.loss
        progress = (self.initial_loss - context.loss) / (self.initial_loss - self.optimal_loss)
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
