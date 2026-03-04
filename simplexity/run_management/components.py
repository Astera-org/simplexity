"""Components for the run."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass
from typing import Any

from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.metrics.metric_tracker import MetricTracker
from simplexity.persistence.model_persister import ModelPersister


@dataclass
class Components:
    """Components for the run."""

    loggers: dict[str, Logger] | None = None
    generative_processes: dict[str, GenerativeProcess] | None = None
    persisters: dict[str, ModelPersister] | None = None
    predictive_models: dict[str, Any] | None = None  # TODO: improve typing
    optimizers: dict[str, Any] | None = None  # TODO: improve typing
    lr_schedulers: dict[str, Any] | None = None  # TODO: improve typing
    metric_trackers: dict[str, MetricTracker] | None = None
    activation_trackers: dict[str, ActivationTracker] | None = None

    def get_logger(self, key: str | None = None) -> Logger | None:
        """Get the logger."""
        return self._get_instance_by_key(self.loggers, key, "logger")

    def get_generative_process(self, key: str | None = None) -> GenerativeProcess | None:
        """Get the generative process."""
        return self._get_instance_by_key(self.generative_processes, key, "generative process")

    def get_persister(self, key: str | None = None) -> ModelPersister | None:
        """Get the persister."""
        return self._get_instance_by_key(self.persisters, key, "persister")

    def get_predictive_model(self, key: str | None = None) -> Any | None:
        """Get the predictive model."""
        return self._get_instance_by_key(self.predictive_models, key, "predictive model")

    def get_optimizer(self, key: str | None = None) -> Any | None:
        """Get the optimizer."""
        return self._get_instance_by_key(self.optimizers, key, "optimizer")

    def get_learning_rate_scheduler(self, key: str | None = None) -> Any | None:
        """Get the learning rate scheduler."""
        return self._get_instance_by_key(self.lr_schedulers, key, "learning rate scheduler")

    def get_metric_tracker(self, key: str | None = None) -> MetricTracker | None:
        """Get the metric tracker."""
        return self._get_instance_by_key(self.metric_trackers, key, "metric tracker")

    def get_activation_tracker(self, key: str | None = None) -> ActivationTracker | None:
        """Get the activation tracker."""
        return self._get_instance_by_key(self.activation_trackers, key, "activation tracker")

    def _get_instance_by_key[T: Any](
        self, instances: dict[str, T] | None, key: str | None, component_name: str
    ) -> T | None:
        """Get the instance by key."""
        if instances is None:
            if key is None:
                return None
            raise KeyError(f"No {component_name} found")
        if key is None:
            if len(instances) == 1:
                return next(iter(instances.values()))
            raise KeyError(f"No key provided and multiple {component_name}s found")
        if key in instances:
            return instances[key]
        ending_matches = [instance_key for instance_key in instances if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return instances[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple {component_name}s with key '{key}' found: {ending_matches}")
        ending_matches = [instance_key for instance_key in instances if instance_key.endswith(f"{key}.instance")]
        if len(ending_matches) == 1:
            return instances[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple {component_name}s with key '{key}.instance' found: {ending_matches}")
        raise KeyError(f"{component_name.capitalize()} with key '{key}' not found")
