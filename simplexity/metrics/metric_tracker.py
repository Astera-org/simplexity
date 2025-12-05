"""Stateful metric tracking for PyTorch training loops.

This module provides a :class:`MetricTracker` that keeps track of
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
    RequiredFields,
    Requirements,
    combine_requirements,
)
from simplexity.utils.torch_nn_utils import extract_learning_rates, snapshot_gradients, snapshot_named_parameters

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
        self.model = model
        self.optimizer = optimizer
        self._metric_groups = self._get_metric_groups(metric_names)
        self._group_requirements = self._get_group_requirements()
        self._warn_missing_context()
        self.context = Context()
        metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self._metrics = self._get_metrics(metric_kwargs)
        self._cache: dict[str, Mapping[str, float]] = {}

    @property
    def metric_groups(self) -> dict[str, list[str]]:
        """Get the metric groups."""
        return self._metric_groups

    def step(self, *, tokens: int | torch.Tensor = 0, loss: float | torch.Tensor = float("inf")) -> None:
        """Advance the global step and update running counters."""
        num_tokens = tokens.numel() if isinstance(tokens, torch.Tensor) else tokens
        loss = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else loss
        self.context = Context(num_tokens=num_tokens, loss=loss)
        self._cache.clear()

        requirements = self._group_requirements[self.step_group].step
        self._update_context(requirements)
        for metric_name in self._metric_groups[self.step_group]:
            metric = self._metrics[metric_name]
            metric.step(self.context)

    def get_metrics(self, group: str = _ALL_GROUP) -> dict[str, float]:
        """Get the metrics for the given group."""
        collected = {}
        requirements = self._group_requirements[group].compute
        self._update_context(requirements)
        for metric_name in self._metric_groups[group]:
            if metric_name not in self._cache:
                metric = self._metrics[metric_name]
                self._cache[metric_name] = metric.compute(self.context)
            collected.update(self._cache[metric_name])
        return collected

    def _get_metric_groups(self, metrics: Mapping[str, Sequence[str]] | Sequence[str] | None) -> dict[str, list[str]]:
        metric_groups: dict[str, list[str]] = {}
        if isinstance(metrics, Mapping):
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

    def _get_group_requirements(self) -> dict[str, Requirements]:
        """Initialize combined Requirements for each metric group."""
        group_requirements: dict[str, Requirements] = {}

        for group, metrics_list in self._metric_groups.items():
            requirements_list = [ALL_METRICS[metric_name].requirements for metric_name in metrics_list]
            group_requirements[group] = combine_requirements(requirements_list)

        return group_requirements

    def _warn_missing_context(self) -> None:
        """Warn if the context is missing required fields."""
        for metric_name in self._metric_groups[self.all_group]:
            requirements = ALL_METRICS[metric_name].requirements
            if self.optimizer is None and requirements.context_field_required("learning_rates"):
                SIMPLEXITY_LOGGER.warning(
                    "[Metrics] %s requires learning rates, but optimizer is not set in MetricTracker", metric_name
                )
            if self.model is None and (
                requirements.context_field_required("gradients")
                or requirements.context_field_required("named_parameters")
            ):
                SIMPLEXITY_LOGGER.warning(
                    "[Metrics] %s requires gradients or named parameters, but model is not set in MetricTracker",
                    metric_name,
                )

    def _get_metrics(self, metric_kwargs: dict[str, Any]) -> dict[str, Metric]:
        requirements = self._group_requirements[self.all_group].init
        self._update_context(requirements)
        return {
            metric_name: ALL_METRICS[metric_name](self.context, **metric_kwargs)
            for metric_name in self._metric_groups[self.all_group]
        }

    def _update_context(self, requirements: RequiredFields) -> None:
        """Update context with required fields for the given group."""
        if (
            self.optimizer is not None
            and getattr(requirements, "learning_rates", False)
            and not self.context.learning_rates
        ):
            self.context.learning_rates = extract_learning_rates(self.optimizer)
        if self.model is not None and getattr(requirements, "gradients", False) and not self.context.gradients:
            self.context.gradients = snapshot_gradients(self.model)
        if (
            self.model is not None
            and getattr(requirements, "named_parameters", False)
            and not self.context.named_parameters
        ):
            self.context.named_parameters = snapshot_named_parameters(self.model)
