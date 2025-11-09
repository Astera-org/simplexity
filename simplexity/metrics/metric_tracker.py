"""Stateful metric tracking for PyTorch training loops.

This module provides a :class:`TrainingMetricTracker` that keeps track of
instantaneous and cumulative metrics derived from optimizer state, running
losses, and snapshots of the model parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol

try:
    import torch
except ImportError as e:  # pragma: no cover - torch is an optional extra
    raise ImportError("To use PyTorch metrics install the torch extra via `uv sync --extra pytorch`.") from e


@dataclass(frozen=True)
class MetricContext:
    """Immutable view of the information required by a metric for one step."""

    model: torch.nn.Module
    step: int
    batch_tokens: int
    total_tokens: int
    learning_rates: Mapping[str, float]
    loss: float
    initial_loss: float | None = None
    optimal_loss: float | None = None
    gradients: Mapping[str, torch.Tensor] | None = None
    named_parameters: Mapping[str, torch.Tensor] | None = None
    initial_named_parameters: Mapping[str, torch.Tensor] | None = None
    previous_named_parameters: Mapping[str, torch.Tensor] | None = None
    parameter_deltas: Mapping[str, torch.Tensor] | None = None


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


class StepMetric:
    """Metric reporting the current optimizer step."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the step metric."""
        return {"step": float(context.step)}


class TokenCountMetric:
    """Tracks instantaneous and cumulative token counts."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the token count metric."""
        return {
            "tokens/batch": float(context.batch_tokens),
            "tokens/total": float(context.total_tokens),
        }


class LearningRateMetric:
    """Reports learning rates for each optimizer param group."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the learning rate metric."""
        values: MutableMapping[str, float] = {}
        for group_name, lr in context.learning_rates.items():
            values[f"lr/{group_name}"] = lr
        if "group_0" in context.learning_rates:
            values["lr"] = context.learning_rates["group_0"]
        return values


class CurrentLossMetric:
    """Logs the instantaneous training loss."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the current loss metric."""
        return {"loss": context.loss}


class LossReferenceMetric:
    """Provides reference loss values if they are available."""

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the loss reference metric."""
        values: MutableMapping[str, float] = {}
        if context.initial_loss is not None:
            values["loss/initial"] = context.initial_loss
            values["loss/delta_from_initial"] = context.loss - context.initial_loss
        if context.optimal_loss is not None:
            values["loss/optimal"] = context.optimal_loss
            values["loss/delta_from_optimal"] = context.loss - context.optimal_loss
        if context.initial_loss is not None and context.optimal_loss is not None:
            denom = context.initial_loss - context.optimal_loss
            if denom != 0:
                values["loss/progress_to_optimal"] = (context.loss - context.optimal_loss) / denom
        return values


class RunningAverageLossMetric:
    """Keeps a running mean of the loss."""

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the running average loss metric."""
        self._total += context.loss
        self._count += 1
        return {"loss/avg": self._total / self._count}


class ParameterNormMetric:
    """Computes the global L2 norm over all parameters."""

    requires_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the parameter norm metric."""
        if context.named_parameters is None:
            raise RuntimeError("ParameterNormMetric requires named parameters")
        return {"params/l2_norm": _tensor_collection_l2_norm(context.named_parameters.values())}


class WeightNormMetric:
    """Computes the L2 norm over parameters whose name ends with 'weight'."""

    requires_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the weight norm metric."""
        if context.named_parameters is None:
            raise RuntimeError("WeightNormMetric requires named parameters")
        weight_tensors = [tensor for name, tensor in context.named_parameters.items() if name.endswith("weight")]
        if not weight_tensors:
            return {"params/weights_l2_norm": 0.0}
        return {"params/weights_l2_norm": _tensor_collection_l2_norm(weight_tensors)}


class DistanceFromInitializationMetric:
    """Reports the parameter space distance from the initial model state."""

    requires_named_parameters = True
    requires_initial_named_parameters = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the distance from initialization metric."""
        if context.named_parameters is None or context.initial_named_parameters is None:
            raise RuntimeError("DistanceFromInitializationMetric requires parameter snapshots")
        distance = _named_tensor_distance(context.named_parameters, context.initial_named_parameters)
        return {"params/distance_from_init": distance}


class GradientNormMetric:
    """Measures gradient magnitudes."""

    requires_gradients = True

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the gradient norm metric."""
        gradients = context.gradients
        if not gradients:
            return {"grads/l2_norm": 0.0}
        return {"grads/l2_norm": _tensor_collection_l2_norm(gradients.values())}


class ParameterUpdateNormMetric:
    """Tracks per-step and cumulative parameter update norms."""

    requires_parameter_deltas = True

    def __init__(self) -> None:
        self._cumulative = 0.0

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the parameter update norm metric."""
        if not context.parameter_deltas:
            return {"params/update_l2_norm": 0.0, "params/update_l2_norm/cumulative": self._cumulative}
        step_norm = _tensor_collection_l2_norm(context.parameter_deltas.values())
        self._cumulative += step_norm
        return {
            "params/update_l2_norm": step_norm,
            "params/update_l2_norm/cumulative": self._cumulative,
        }


class PeakLearningRateMetric:
    """Captures instantaneous and peak learning-rate statistics."""

    def __init__(self, schedule_fn: Callable[[int], float] | None = None) -> None:
        self._peak_lr = 0.0
        self._peak_weighted = 0.0
        self._schedule_fn = schedule_fn

    def compute(self, context: MetricContext) -> Mapping[str, float]:
        """Compute the peak learning rate metric."""
        current_lr = max(context.learning_rates.values(), default=0.0)
        self._peak_lr = max(self._peak_lr, current_lr)
        weighted = current_lr * float(context.batch_tokens)
        self._peak_weighted = max(self._peak_weighted, weighted)
        values: MutableMapping[str, float] = {
            "lr/current_peak": self._peak_lr,
            "lr/weighted_by_tokens": weighted,
            "lr/weighted_peak": self._peak_weighted,
            "lr/weighted_normalized": weighted / float(context.batch_tokens or 1),
        }
        if self._schedule_fn:
            scheduled_lr = float(self._schedule_fn(context.step))
            values["lr/schedule"] = scheduled_lr
            if scheduled_lr:
                values["lr/relative_to_schedule"] = current_lr / scheduled_lr
        return values


class TrainingMetricTracker:
    """Stateful helper that orchestrates instantaneous and cumulative metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        metrics: Sequence[TrainingMetric] | None = None,
        initial_loss: float | None = None,
        optimal_loss: float | None = None,
        lr_schedule_fn: Callable[[int], float] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.initial_loss = initial_loss
        self.optimal_loss = optimal_loss
        self.lr_schedule_fn = lr_schedule_fn
        self._total_tokens = 0
        self._metrics = list(metrics) if metrics is not None else self._default_metrics()
        self._set_requirement_flags()
        initial_snapshot: Mapping[str, torch.Tensor] | None = None
        if self._needs_initial_named_parameters or self._needs_parameter_deltas:
            initial_snapshot = self._snapshot_named_parameters(model)
        self._initial_named_parameters = initial_snapshot if self._needs_initial_named_parameters else None
        self._previous_named_parameters = initial_snapshot if self._needs_parameter_deltas else None

    @staticmethod
    def _snapshot_named_parameters(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        return {name: param.detach().clone().cpu() for name, param in model.named_parameters()}

    @staticmethod
    def _gather_named_parameters(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        return {name: param.detach().clone().cpu() for name, param in model.named_parameters()}

    @staticmethod
    def _gather_gradients(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        gradients: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone().cpu()
        return gradients

    @staticmethod
    def _compute_parameter_deltas(
        current: Mapping[str, torch.Tensor], previous: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        deltas: dict[str, torch.Tensor] = {}
        for name, tensor in current.items():
            prev_tensor = previous.get(name)
            if prev_tensor is None:
                continue
            deltas[name] = tensor - prev_tensor
        return deltas

    def _set_requirement_flags(self) -> None:
        def requires(attribute: str) -> bool:
            return any(bool(getattr(metric, attribute, False)) for metric in self._metrics)

        self._needs_parameter_deltas = requires("requires_parameter_deltas")
        self._needs_gradients = requires("requires_gradients")
        self._needs_initial_named_parameters = requires("requires_initial_named_parameters")
        self._needs_named_parameters = self._needs_parameter_deltas or requires("requires_named_parameters")

    def _default_metrics(self) -> list[TrainingMetric]:
        return [
            StepMetric(),
            TokenCountMetric(),
            LearningRateMetric(),
            CurrentLossMetric(),
            LossReferenceMetric(),
            RunningAverageLossMetric(),
            ParameterNormMetric(),
            WeightNormMetric(),
            DistanceFromInitializationMetric(),
            GradientNormMetric(),
            ParameterUpdateNormMetric(),
            PeakLearningRateMetric(schedule_fn=self.lr_schedule_fn),
        ]

    def record_initial_loss(self, loss: float) -> None:
        """Persist the loss observed by the randomly initialized model."""
        if self.initial_loss is None:
            self.initial_loss = float(loss)

    def update(
        self,
        *,
        step: int,
        loss: float | torch.Tensor,
        tokens_in_batch: int,
        learning_rates: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Update all metrics for the provided training step."""
        loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        self._total_tokens += int(tokens_in_batch)
        named_parameters = self._gather_named_parameters(self.model) if self._needs_named_parameters else None
        gradients = self._gather_gradients(self.model) if self._needs_gradients else None
        previous_named_parameters = self._previous_named_parameters if self._needs_parameter_deltas else None
        parameter_deltas = None
        if self._needs_parameter_deltas and named_parameters is not None and previous_named_parameters is not None:
            parameter_deltas = self._compute_parameter_deltas(named_parameters, previous_named_parameters)
        context = MetricContext(
            step=step,
            batch_tokens=int(tokens_in_batch),
            total_tokens=self._total_tokens,
            loss=loss_value,
            learning_rates=learning_rates or self._extract_learning_rates(),
            model=self.model,
            named_parameters=named_parameters,
            initial_named_parameters=self._initial_named_parameters,
            previous_named_parameters=previous_named_parameters,
            initial_loss=self.initial_loss,
            optimal_loss=self.optimal_loss,
            gradients=gradients,
            parameter_deltas=parameter_deltas,
        )
        metrics: dict[str, float] = {}
        for metric in self._metrics:
            metrics.update(metric.compute(context))
        if self._needs_parameter_deltas and named_parameters is not None:
            self._previous_named_parameters = named_parameters
        return metrics

    def _extract_learning_rates(self) -> Mapping[str, float]:
        rates: dict[str, float] = {}
        for idx, group in enumerate(self.optimizer.param_groups):
            lr = float(group.get("lr", 0.0))
            rates[f"group_{idx}"] = lr
        return rates


__all__ = [
    "MetricContext",
    "TrainingMetric",
    "TrainingMetricTracker",
    "StepMetric",
    "TokenCountMetric",
    "LearningRateMetric",
    "CurrentLossMetric",
    "LossReferenceMetric",
    "RunningAverageLossMetric",
    "ParameterNormMetric",
    "WeightNormMetric",
    "DistanceFromInitializationMetric",
    "GradientNormMetric",
    "ParameterUpdateNormMetric",
    "PeakLearningRateMetric",
]
