"""Stateful metric tracking for PyTorch training loops.

This module provides a :class:`TrainingMetricTracker` that keeps track of
instantaneous and cumulative metrics derived from optimizer state, running
losses, and snapshots of the model parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from simplexity.metrics.metrics import (
    ALL_METRICS,
    Context,
    Metric,
    Requirements,
    combine_requirements,
)

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")

_ALL_GROUP = "all"
_STEP_GROUP = "step"


class MetricTracker:  # pylint: disable=too-many-instance-attributes
    """Stateful helper that orchestrates instantaneous and cumulative metrics."""

    all_group: str = _ALL_GROUP
    step_group: str = _STEP_GROUP

    def __init__(  # pylint: disable=too-many-arguments
        self,
        metric_names: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._metric_groups = self._initialize_metric_groups(metric_names)
        self.model = model
        self.optimizer = optimizer
        self._context = Context()
        self._group_requirements = self._compute_group_requirements()
        metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self._metrics = self._initialize_metrics(metric_kwargs)
        self._cache: dict[str, Mapping[str, float]] = {}

    def step(self, *, tokens: int | torch.Tensor, loss: float | torch.Tensor) -> None:
        """Advance the global step and update running counters."""
        self._context = Context(
            step=self._context.step + 1,
            num_tokens=tokens.numel() if isinstance(tokens, torch.Tensor) else tokens,
            loss=float(loss),
        )
        self._cache.clear()

        requirements = self._group_requirements[self.step_group].step
        self._context = self._update_context(requirements)
        for metric_name in self._metric_groups[self.step_group]:
            metric = self._metrics[metric_name]
            metric.step(self._context)

    def get_metrics(self, group: str = _ALL_GROUP) -> dict[str, float]:
        """Get the metrics for the given group."""
        collected = {}
        requirements = self._group_requirements[group].compute
        self._context = self._update_context(requirements)
        for metric_name in self._metric_groups[group]:
            if metric_name not in self._cache:
                metric = self._metrics[metric_name]
                self._cache[metric_name] = metric.compute(self._context)
            collected.update(self._cache[metric_name])
        return collected

    def _initialize_metric_groups(
        self, metrics: Mapping[str, Sequence[str]] | Sequence[str] | None
    ) -> dict[str, list[str]]:
        metric_groups: dict[str, list[str]] = {}
        if isinstance(metrics, dict):
            metric_groups = {group: list(metrics_list) for group, metrics_list in metrics.items()}
            all_metric_names = list(
                set([metric_name for metrics_list in metric_groups.values() for metric_name in metrics_list])
            )
            metric_groups[self.all_group] = all_metric_names
        elif isinstance(metrics, Sequence):
            metric_groups = {self.all_group: list(set(metrics))}
        else:
            metric_groups = {self.all_group: list(ALL_METRICS.keys())}

        def requires_update_every_step(metric_name: str) -> bool:
            metric_class = ALL_METRICS[metric_name]
            return metric_class.requirements.step_required

        metric_groups[self.step_group] = [
            metric_name for metric_name in metric_groups[self.all_group] if requires_update_every_step(metric_name)
        ]
        return metric_groups

    def _compute_group_requirements(self) -> dict[str, Requirements]:
        """Compute combined Requirements for each metric group."""
        group_requirements: dict[str, Requirements] = {}

        for group, metrics_list in self._metric_groups.items():
            requirements_list = [ALL_METRICS[metric_name].requirements for metric_name in metrics_list]
            group_requirements[group] = combine_requirements(requirements_list)

        return group_requirements

    def _initialize_metrics(self, metric_kwargs: dict[str, Any]) -> dict[str, Metric]:
        requirements = self._group_requirements[self.all_group].init
        self._context = self._update_context(requirements)
        return {
            metric_name: ALL_METRICS[metric_name](self._context, **metric_kwargs)
            for metric_name in self._metric_groups[self.all_group]
        }

    def _update_context(self, requirements: Requirements) -> Context:
        """Update context with required fields for the given group."""
        if self._context.learning_rates is None and getattr(requirements, "learning_rates", False):
            self._context.learning_rates = self._extract_learning_rates()
        if self._context.gradients is None and getattr(requirements, "gradients", False):
            self._context.gradients = self._snapshot_gradients()
        if self._context.named_parameters is None and getattr(requirements, "named_parameters", False):
            self._context.named_parameters = self._snapshot_named_parameters()
        return self._context

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
                gradients[name] = param.grad.detach().clone()
        return gradients

    def _snapshot_named_parameters(self) -> Mapping[str, torch.Tensor]:
        assert self.model is not None, "Model is required for metrics that require named parameters"
        return {name: param.detach().clone() for name, param in self.model.named_parameters()}
