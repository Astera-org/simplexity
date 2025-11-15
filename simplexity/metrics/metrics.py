"""Metrics for tracking training progress."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Protocol

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
    previous_named_parameters: Mapping[str, torch.Tensor] | None = None
    current_named_parameters: Mapping[str, torch.Tensor] | None = None


class TrainingMetric(Protocol):
    """Protocol for metrics that can be plugged into the tracker."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:  # pragma: no cover - Protocol
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

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the token count metric."""
        return {
            "tokens/batch": float(context.batch_tokens),
            "tokens/total": float(context.total_tokens),
        }


class LearningRateMetric:
    """Reports learning rates for each optimizer param group."""

    requires_learning_rates = True

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
    cumulative: float = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the learning rate weighted tokens metric."""
        weighted_tokens = context.learning_rates["lr"] * float(context.batch_tokens)
        self.cumulative += weighted_tokens
        return {
            "tokens/lr_weighted": weighted_tokens,
            "tokens/lr_weighted/cumulative": self.cumulative,
        }


class GradientWeightedTokensMetric:
    """Tracks the gradient weighted tokens."""

    requires_learning_rates = True
    requires_gradients = True
    cumulative: float = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the gradient weighted tokens metric."""
        assert context.gradients is not None, "Gradients are required for this metric"
        gradient_norm = _tensor_collection_l2_norm(context.gradients.values())
        weighted_tokens = context.learning_rates["lr"] * gradient_norm * float(context.batch_tokens)
        self.cumulative += weighted_tokens
        return {
            "tokens/gradient_weighted": weighted_tokens,
            "tokens/gradient_weighted/cumulative": self.cumulative,
        }


class CurrentLossMetric:
    """Logs the instantaneous training loss."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the current loss metric."""
        return {"loss": context.loss}


class ParameterNormMetric:
    """Computes the global L2 norm over all parameters."""

    requires_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the parameter norm metric."""
        assert context.current_named_parameters is not None, "Current named parameters are required for this metric"
        return {"params/l2_norm": _tensor_collection_l2_norm(context.current_named_parameters.values())}


class WeightNormMetric:
    """Computes the L2 norm over parameters whose name ends with 'weight'."""

    requires_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the weight norm metric."""
        assert context.current_named_parameters is not None, "Current named parameters are required for this metric"
        weight_tensors = [
            tensor for name, tensor in context.current_named_parameters.items() if name.endswith("weight")
        ]
        return {"params/weights_l2_norm": _tensor_collection_l2_norm(weight_tensors)}


class DistanceFromInitializationMetric:
    """Reports the parameter space distance from the initial model state."""

    initial_named_parameters: Mapping[str, torch.Tensor]
    requires_named_parameters = True
    requires_initial_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the distance from initialization metric."""
        assert context.current_named_parameters is not None, "Current named parameters are required for this metric"
        distance = _named_tensor_distance(context.current_named_parameters, self.initial_named_parameters)
        return {"params/distance_from_init": distance}


class CumulativeParameterUpdateMetric:
    """Tracks the cumulative parameter update."""

    requires_current_named_parameters = True
    requires_previous_named_parameters = True
    cumulative: float = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the update norm metric."""
        assert context.current_named_parameters is not None, "Current named parameters are required for this metric"
        assert context.previous_named_parameters is not None, "Previous named parameters are required for this metric"
        step_norm = _named_tensor_distance(context.current_named_parameters, context.previous_named_parameters)
        self.cumulative += step_norm
        return {
            "params/update_l2_norm": step_norm,
            "params/update_l2_norm/cumulative": self.cumulative,
        }


class FisherInformationMetric:
    """Tracks the Fisher information."""

    requires_gradients = True
    cumulative: float = 0.0

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

    initial_loss: float
    optimal_loss: float
    requires_initial_loss = True
    requires_optimal_loss = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the loss progress metric."""
        progress = (self.initial_loss - context.loss) / (self.initial_loss - self.optimal_loss)
        return {"loss/progress_to_optimal": progress}


ALL_METRICS = {
    "tokens/batch": TokensMetric(),
    "tokens/total": TokensMetric(),
    "lr": LearningRateMetric(),
    "loss": CurrentLossMetric(),
    "params/distance_from_init": DistanceFromInitializationMetric(),
    "params/update_l2_norm": CumulativeParameterUpdateMetric(),
    "params/fisher_information": FisherInformationMetric(),
    "loss/progress_to_optimal": LossProgressMetric(),
    "params/l2_norm": ParameterNormMetric(),
    "params/weights_l2_norm": WeightNormMetric(),
    "tokens/lr_weighted": LearningRateWeightedTokensMetric(),
    "tokens/gradient_weighted": GradientWeightedTokensMetric(),
}
