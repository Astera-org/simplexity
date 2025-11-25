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
    Metric,
    MetricContext,
)

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


class MetricTracker:  # pylint: disable=too-many-instance-attributes
    """Stateful helper that orchestrates instantaneous and cumulative metrics."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        metric_names: dict[str, Sequence[str]] | Sequence[str] | None = None,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if metric_kwargs is None:
            metric_kwargs = {}
        self._metric_groups = self._initialize_metric_groups(metric_names)
        self._missing_update_keys = self._set_missing_update_keys()
        self._needs_learning_rates = self._set_requirement_flag("requires_learning_rates")
        self._needs_gradients = self._set_requirement_flag("requires_gradients")
        self._needs_named_parameters = self._set_requirement_flag("requires_named_parameters")

        self.model = model

        named_parameters: Mapping[str, torch.Tensor] | None = None
        if self._needs_named_parameters:
            named_parameters = self._snapshot_named_parameters()
            metric_kwargs["named_parameters"] = named_parameters

        self.optimizer = optimizer
        self._context = self._initialize_context(named_parameters)
        self._metrics = self._initialize_metrics(metric_kwargs)

    def metrics(self, group: str = "all") -> dict[str, float]:
        """Get the metrics for the given group."""
        collected = {}
        for metric_name in self._metric_groups[group]:
            computed = self._metrics[metric_name].compute()
            collected.update(computed)
        return collected

    def step(
        self,
        *,
        tokens: int | torch.Tensor,
        loss: float | torch.Tensor,
    ) -> None:
        """Advance the global step and update running counters."""
        self._context.step += 1
        num_tokens = tokens.numel() if isinstance(tokens, torch.Tensor) else tokens
        self._context.num_tokens = num_tokens
        self._context.loss = float(loss) if isinstance(loss, torch.Tensor) else loss

    def update_metrics(self, group: str = "all") -> None:
        """Update the metrics for the given group."""
        if group in self._missing_update_keys:
            SIMPLEXITY_LOGGER.warning("Update of metric group %s misses metrics that require updates every step", group)
        
        if self._needs_learning_rates[group]:
            self._context.learning_rates = self._extract_learning_rates()
        if self._needs_gradients[group]:
            self._context.gradients = self._snapshot_gradients()
        if self._needs_named_parameters[group]:
            self._context.named_parameters = self._snapshot_named_parameters()
            
        for metric_name in self._metric_groups[group]:
            metric = self._metrics[metric_name]
            metric.update(self._context)

    def update(
        self,
        group: str = "all",
        *,
        tokens: int | torch.Tensor,
        loss: float | torch.Tensor,
    ) -> None:
        """Update the metric tracker with the given context (Deprecated).
        
        This method combines step() and update_metrics() for backward compatibility.
        """
        self.step(tokens=tokens, loss=loss)
        self.update_metrics(group=group)

    def _initialize_metric_groups(
        self, metrics: dict[str, Sequence[str]] | Sequence[str] | None
    ) -> dict[str, list[str]]:
        if isinstance(metrics, dict):
            metric_groups = {group: list(metrics_list) for group, metrics_list in metrics.items()}
            all_metric_names = list(
                set([metric_name for metrics_list in metric_groups.values() for metric_name in metrics_list])
            )
            metric_groups["all"] = all_metric_names
            return metric_groups
        if isinstance(metrics, Sequence):
            return {"all": list(metrics)}
        metric_groups = {"all": list(ALL_METRICS.keys())}

        def requires_update_every_step(metric_name: str) -> bool:
            return bool(getattr(ALL_METRICS[metric_name], "update_every_step", False))

        flat_metric_names = list(
            set(
                [
                    metric_name
                    for group in metric_groups.values()
                    for metric_name in group
                    if requires_update_every_step(metric_name)
                ]
            )
        )
        metric_groups["update_every_step"] = flat_metric_names
        return metric_groups

    def _set_missing_update_keys(self) -> set[str]:
        missing_update_keys = set()
        required_metric_names = set(self._metric_groups["update_every_step"])
        for group, metrics_list in self._metric_groups.items():
            if required_metric_names.difference(set(metrics_list)):
                missing_update_keys.add(group)
        return missing_update_keys

    def _set_requirement_flag(self, attribute: str) -> dict[str, bool]:
        def requires(metrics_list: list[str]) -> bool:
            return any(bool(getattr(ALL_METRICS[metric], attribute, False)) for metric in metrics_list)

        return {group: requires(metrics_list) for group, metrics_list in self._metric_groups.items()}

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

    def _initialize_context(self, named_parameters: Mapping[str, torch.Tensor] | None = None) -> MetricContext:
        learning_rates: Mapping[str, float] = {}
        if self._needs_learning_rates:
            learning_rates = self._extract_learning_rates()

        gradients: Mapping[str, torch.Tensor] | None = None
        if self._needs_gradients:
            assert self.model is not None, "Model is required for metrics that require gradients"
            gradients = self._snapshot_gradients()

        return MetricContext(
            step=0,
            num_tokens=0,
            loss=float("inf"),
            learning_rates=learning_rates,
            gradients=gradients,
            named_parameters=named_parameters,
        )

    def _initialize_metrics(self, metric_kwargs: dict[str, Any]) -> dict[str, Metric]:
        flat_metric_names = list(set([metric_name for group in self._metric_groups.values() for metric_name in group]))
        return {metric_name: ALL_METRICS[metric_name](**metric_kwargs) for metric_name in flat_metric_names}
