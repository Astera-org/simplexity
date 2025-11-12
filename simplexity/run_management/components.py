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

import jax

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister


@dataclass
class Components:
    """Components for the run."""

    loggers: dict[str, Logger] | None = None
    generative_processes: dict[str, GenerativeProcess] | None = None
    initial_states: dict[str, jax.Array] | None = None
    persisters: dict[str, ModelPersister] | None = None
    predictive_models: dict[str, Any] | None = None  # TODO: improve typing
    optimizers: dict[str, Any] | None = None  # TODO: improve typing

    def get_logger(self, key: str | None = None) -> Logger | None:
        """Get the logger."""
        if self.loggers is None:
            if key is None:
                return None
            raise KeyError("No loggers found")
        if key is None:
            if len(self.loggers) == 1:
                return next(iter(self.loggers.values()))
            raise KeyError("No key provided and multiple loggers found")
        if key in self.loggers:
            return self.loggers[key]
        ending_matches = [instance_key for instance_key in self.loggers if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.loggers[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple loggers with key '{key}' found: {ending_matches}")
        raise KeyError(f"Logger with key '{key}' not found")

    def get_generative_process(self, key: str | None = None) -> GenerativeProcess | None:
        """Get the generative process."""
        if self.generative_processes is None:
            if key is None:
                return None
            raise KeyError("No generative processes found")
        if key is None:
            if len(self.generative_processes) == 1:
                return next(iter(self.generative_processes.values()))
            raise KeyError("No key provided and multiple generative processes found")
        if key in self.generative_processes:
            return self.generative_processes[key]
        ending_matches = [instance_key for instance_key in self.generative_processes if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.generative_processes[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple generative processes with key '{key}' found: {ending_matches}")
        raise KeyError(f"Generative process with key '{key}' not found")

    def get_initial_state(self, key: str | None = None) -> jax.Array | None:
        """Get the initial state."""
        if self.initial_states is None:
            if key is None:
                return None
            raise KeyError("No initial states found")
        if key is None:
            if len(self.initial_states) == 1:
                return next(iter(self.initial_states.values()))
            raise KeyError("No key provided and multiple initial states found")
        if key in self.initial_states:
            return self.initial_states[key]
        ending_matches = [instance_key for instance_key in self.initial_states if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.initial_states[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple initial states with key '{key}' found: {ending_matches}")
        raise KeyError(f"Initial state with key '{key}' not found")

    def get_persister(self, key: str | None = None) -> ModelPersister | None:
        """Get the persister."""
        if self.persisters is None:
            if key is None:
                return None
            raise KeyError("No persisters found")
        if key is None:
            if len(self.persisters) == 1:
                return next(iter(self.persisters.values()))
            raise KeyError("No key provided and multiple persisters found")
        if key in self.persisters:
            return self.persisters[key]
        ending_matches = [instance_key for instance_key in self.persisters if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.persisters[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple persisters with key '{key}' found: {ending_matches}")
        raise KeyError(f"Persister with key '{key}' not found")

    def get_predictive_model(self, key: str | None = None) -> Any | None:
        """Get the predictive model."""
        if self.predictive_models is None:
            if key is None:
                return None
            raise KeyError("No predictive models found")
        if key is None:
            if len(self.predictive_models) == 1:
                return next(iter(self.predictive_models.values()))
            raise KeyError("No key provided and multiple predictive models found")
        if key in self.predictive_models:
            return self.predictive_models[key]
        ending_matches = [instance_key for instance_key in self.predictive_models if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.predictive_models[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple predictive models with key '{key}' found: {ending_matches}")
        raise KeyError(f"Predictive model with key '{key}' not found")

    def get_optimizer(self, key: str | None = None) -> Any | None:
        """Get the optimizer."""
        if self.optimizers is None:
            if key is None:
                return None
            raise KeyError("No optimizers found")
        if key is None:
            if len(self.optimizers) == 1:
                return next(iter(self.optimizers.values()))
            raise KeyError("No key provided and multiple optimizers found")
        if key in self.optimizers:
            return self.optimizers[key]
        ending_matches = [instance_key for instance_key in self.optimizers if instance_key.endswith(key)]
        if len(ending_matches) == 1:
            return self.optimizers[ending_matches[0]]
        if len(ending_matches) > 1:
            raise KeyError(f"Multiple optimizers with key '{key}' found: {ending_matches}")
        raise KeyError(f"Optimizer with key '{key}' not found")
