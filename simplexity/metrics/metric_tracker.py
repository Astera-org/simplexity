"""Stateful metric tracking for PyTorch training loops.

This module provides a :class:`TrainingMetricTracker` that keeps track of
instantaneous and cumulative metrics derived from optimizer state, running
losses, and snapshots of the model parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from simplexity.metrics.metrics import (
    ALL_METRICS,
    MetricContext,
    TrainingMetric,
)


class TrainingMetricTracker:  # pylint: disable=too-many-instance-attributes
    """Stateful helper that orchestrates instantaneous and cumulative metrics."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        metric_names: dict[str, Sequence[str]] | Sequence[str] | None = None,
        *,
        initial_loss: float,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        optimal_loss: float | None = None,
    ) -> None:
        self._set_requirement_flags(metric_names)

        self.model = model

        initial_named_parameters: Mapping[str, torch.Tensor] | None = None
        if self._needs_named_parameters:
            initial_named_parameters = self._snapshot_named_parameters()

        self.optimizer = optimizer
        self._context = self._initialize_context(initial_loss, initial_named_parameters)
        self._metrics = self._initialize_metrics(metric_names, initial_loss, optimal_loss, initial_named_parameters)
        self._metric_groups = self._initialize_metric_groups(metric_names)

    def metrics(self, group: str = "all") -> dict[str, float]:
        """Get the metrics for the given group."""
        collected = {}
        for metric_name in self._metric_groups[group]:
            computed = self._metrics[metric_name].compute(self._context)
            collected.update(computed)
        return collected

    def update(
        self,
        *,
        batch_tokens: int | torch.Tensor,
        loss: float | torch.Tensor,
    ) -> None:
        """Update the metric tracker with the given context."""
        self._context.step += 1
        num_batch_tokens = batch_tokens.numel() if isinstance(batch_tokens, torch.Tensor) else batch_tokens
        self._context.batch_tokens = num_batch_tokens
        self._context.total_tokens += num_batch_tokens
        self._context.loss = float(loss) if isinstance(loss, torch.Tensor) else loss
        if self._needs_learning_rates:
            self._context.learning_rates = self._extract_learning_rates()
        if self._needs_gradients:
            self._context.gradients = self._snapshot_gradients()
        if self._needs_previous_named_parameters:
            self._context.previous_named_parameters = self._context.current_named_parameters
        if self._needs_current_named_parameters:
            self._context.current_named_parameters = self._snapshot_named_parameters()

    def _set_requirement_flags(self, metric_names: dict[str, Sequence[str]] | Sequence[str] | None) -> None:
        metrics_list: list[str] = []
        if metric_names is None:
            metrics_list = list(ALL_METRICS.keys())
        elif isinstance(metric_names, dict):
            metrics_list = [metric for group in metric_names.values() for metric in group]
        elif isinstance(metric_names, Sequence):
            metrics_list = list(metric_names)

        def requires(attribute: str) -> bool:
            return any(bool(getattr(ALL_METRICS[metric], attribute, False)) for metric in metrics_list)

        self._needs_learning_rates = requires("requires_learning_rates")
        self._needs_gradients = requires("requires_gradients")
        self._needs_previous_named_parameters = requires("requires_previous_named_parameters")
        self._needs_current_named_parameters = self._needs_previous_named_parameters or requires(
            "requires_current_named_parameters"
        )
        self._needs_initial_named_parameters = requires("requires_initial_named_parameters")
        self._needs_named_parameters = (
            self._needs_current_named_parameters
            or self._needs_initial_named_parameters
            or self._needs_previous_named_parameters
        )

    def _extract_learning_rates(self) -> Mapping[str, float]:
        assert self.optimizer is not None, "Optimizer is required for metrics that require learning rates"
        rates: dict[str, float] = {}
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{idx}")
            lr = float(group.get("lr", 0.0))
            rates[name] = lr
        return rates

    def _snapshot_gradients(self) -> Mapping[str, torch.Tensor]:
        assert self.model is not None, "Model is required for metrics that require gradients"
        gradients: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone().cpu()
        return gradients

    def _snapshot_named_parameters(self) -> Mapping[str, torch.Tensor]:
        assert self.model is not None, "Model is required for metrics that require named parameters"
        return {name: param.detach().clone().cpu() for name, param in self.model.named_parameters()}

    def _initialize_context(
        self, initial_loss: float, initial_named_parameters: Mapping[str, torch.Tensor] | None = None
    ) -> MetricContext:
        learning_rates: Mapping[str, float] = {}
        if self._needs_learning_rates:
            learning_rates = self._extract_learning_rates()

        gradients: Mapping[str, torch.Tensor] | None = None
        if self._needs_gradients:
            assert self.model is not None, "Model is required for metrics that require gradients"
            gradients = self._snapshot_gradients()

        current_named_parameters = initial_named_parameters if self._needs_named_parameters else None
        previous_named_parameters = initial_named_parameters if self._needs_previous_named_parameters else None

        return MetricContext(
            step=0,
            batch_tokens=0,
            total_tokens=0,
            loss=initial_loss,
            learning_rates=learning_rates,
            gradients=gradients,
            current_named_parameters=current_named_parameters,
            previous_named_parameters=previous_named_parameters,
        )

    def _initialize_metrics(
        self,
        metric_names: dict[str, Sequence[str]] | Sequence[str] | None,
        initial_loss: float | None,
        optimal_loss: float | None,
        initial_named_parameters: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, TrainingMetric]:
        def requires(metric: TrainingMetric, attribute: str) -> bool:
            return bool(getattr(metric, attribute, False))

        def initialize_metric(metric_name: str) -> TrainingMetric:
            kwargs: dict[str, Any] = {}
            metric = ALL_METRICS[metric_name]
            if requires(metric, "requires_initial_named_parameters"):
                assert initial_named_parameters is not None, f"Initial named parameters are required for {metric_name}"
                kwargs["initial_named_parameters"] = initial_named_parameters
            if requires(metric, "requires_initial_loss"):
                assert initial_loss is not None, f"Initial loss is required for {metric_name}"
                kwargs["initial_loss"] = initial_loss
            if requires(metric, "requires_optimal_loss"):
                assert optimal_loss is not None, f"Optimal loss is required for {metric_name}"
                kwargs["optimal_loss"] = optimal_loss
            return metric(**kwargs)

        flat_metric_names: list[str] = []
        if metric_names is None:
            flat_metric_names = list(ALL_METRICS.keys())
        elif isinstance(metric_names, dict):
            flat_metric_names = list(set([metric_name for group in metric_names.values() for metric_name in group]))
        elif isinstance(metric_names, Sequence):
            flat_metric_names = list(metric_names)
        return {metric_name: initialize_metric(metric_name) for metric_name in flat_metric_names}

    def _initialize_metric_groups(
        self, metrics: dict[str, Sequence[str]] | Sequence[str] | None
    ) -> dict[str, list[str]]:
        if isinstance(metrics, dict):
            metric_groups = {name: list(group) for name, group in metrics.items()}
            all_metric_names = list(set([metric_name for group in metric_groups.values() for metric_name in group]))
            metric_groups["all"] = all_metric_names
            return metric_groups
        if isinstance(metrics, Sequence):
            return {"all": list(metrics)}
        return {"all": list(ALL_METRICS.keys())}
