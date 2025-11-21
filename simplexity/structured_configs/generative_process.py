"""Generative process configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, _validate_instance_config
from simplexity.structured_configs.validation import (
    _validate_bool,
    _validate_initial_state,
    _validate_mapping,
    _validate_non_negative_float,
    _validate_non_negative_int,
    _validate_nonempty_str,
    _validate_positive_int,
    _validate_sequence,
    _validate_transition_matrices,
)
from simplexity.utils.config_utils import dynamic_resolve

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


@dataclass
class HiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the hidden markov model builder."""

    process_name: str
    process_params: Mapping[str, Any] | None = None
    initial_state: jax.Array | Sequence[float] | None = None

    def __init__(
        self,
        process_name: str,
        process_params: Mapping[str, Any] | None = None,
        initial_state: jax.Array | Sequence[float] | None = None,
        _target_: str = "simplexity.generative_processes.builder.build_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_name = process_name
        self.process_params = process_params
        self.initial_state = initial_state


def is_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_hidden_markov_model"


def is_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_hidden_markov_model_builder_target(target)
    return False


def validate_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a HiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with HiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_name = cfg.get("process_name")
    process_params = cfg.get("process_params")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(cfg, expected_target="simplexity.generative_processes.builder.build_hidden_markov_model")
    _validate_nonempty_str(process_name, "HiddenMarkovModelBuilderInstanceConfig.process_name")
    _validate_mapping(
        process_params, "HiddenMarkovModelBuilderInstanceConfig.process_params", key_type=str, is_none_allowed=True
    )
    _validate_sequence(
        initial_state, "HiddenMarkovModelBuilderInstanceConfig.initial_state", element_type=float, is_none_allowed=True
    )


@dataclass
class GeneralizedHiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the generalized hidden markov model builder."""

    process_name: str
    process_params: Mapping[str, Any] | None = None
    initial_state: jax.Array | Sequence[float] | None = None

    def __init__(
        self,
        process_name: str,
        process_params: Mapping[str, Any] | None = None,
        initial_state: jax.Array | Sequence[float] | None = None,
        _target_: str = "simplexity.generative_processes.builder.build_generalized_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_name = process_name
        self.process_params = process_params
        self.initial_state = initial_state


def is_generalized_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a generalized hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_generalized_hidden_markov_model"


def is_generalized_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generalized hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generalized_hidden_markov_model_builder_target(target)
    return False


def validate_generalized_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a GeneralizedHiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with GeneralizedHiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_name = cfg.get("process_name")
    process_params = cfg.get("process_params")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.builder.build_generalized_hidden_markov_model"
    )
    _validate_nonempty_str(process_name, "GeneralizedHiddenMarkovModelBuilderInstanceConfig.process_name")
    _validate_mapping(
        process_params,
        "GeneralizedHiddenMarkovModelBuilderInstanceConfig.process_params",
        key_type=str,
        is_none_allowed=True,
    )
    _validate_sequence(
        initial_state,
        "GeneralizedHiddenMarkovModelBuilderInstanceConfig.initial_state",
        element_type=float,
        is_none_allowed=True,
    )


@dataclass
class NonergodicHiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the nonergodic hidden markov model builder."""

    process_names: list[str]
    process_params: Sequence[Mapping[str, Any]]
    process_weights: Sequence[float]
    vocab_maps: Sequence[Sequence[int]] | None = None
    add_bos_token: bool = False

    def __init__(
        self,
        process_names: list[str],
        process_params: Sequence[Mapping[str, Any]],
        process_weights: Sequence[float],
        vocab_maps: Sequence[Sequence[int]] | None = None,
        add_bos_token: bool = False,
        _target_: str = "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_names = process_names
        self.process_params = process_params
        self.process_weights = process_weights
        self.vocab_maps = vocab_maps
        self.add_bos_token = add_bos_token


def is_nonergodic_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a nonergodic hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model"


def is_nonergodic_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a nonergodic hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_nonergodic_hidden_markov_model_builder_target(target)
    return False


def validate_nonergodic_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a NonergodicHiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with NonergodicHiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_names = cfg.get("process_names")
    process_params = cfg.get("process_params")
    process_weights = cfg.get("process_weights")
    vocab_maps = cfg.get("vocab_maps")
    add_bos_token = cfg.get("add_bos_token")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model"
    )
    if not isinstance(process_names, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names must be a list, got {type(process_names)}"
        )
    if not isinstance(process_params, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_params must be a sequence, "
            f"got {type(process_params)}"
        )
    if not isinstance(process_weights, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_weights must be a sequence, "
            f"got {type(process_weights)}"
        )
    if vocab_maps is None:
        _vocab_maps = [None] * len(process_names)
    else:
        if not isinstance(vocab_maps, Sequence):
            raise ConfigValidationError(
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.vocab_maps must be a sequence, "
                f"got {type(vocab_maps)}"
            )
        _vocab_maps = vocab_maps
    try:
        for i, (name, params, weights, vocab_map) in enumerate(
            zip(process_names, process_params, process_weights, _vocab_maps, strict=True)
        ):
            _validate_nonempty_str(name, f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names[{i}]")
            _validate_mapping(
                params,
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_params[{i}]",
                key_type=str,
                is_none_allowed=True,
            )
            _validate_non_negative_float(
                weights, f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_weights[{i}]"
            )
            _validate_sequence(
                vocab_map,
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.vocab_maps[{i}]",
                element_type=int,
                is_none_allowed=True,
            )
    except ValueError as e:
        var_str = "and process_weights" if vocab_maps is not None else "process_weights, and vocab_maps"
        vocab_len_str = f"!= {len(vocab_maps)}" if vocab_maps is not None else ""
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names, process_params, {var_str}"
            f"must have the same length, "
            f"got {len(process_names)} != {len(process_params)} != {len(process_weights)}{vocab_len_str}"
        ) from e
    _validate_bool(
        add_bos_token, "NonergodicHiddenMarkovModelBuilderInstanceConfig.add_bos_token", is_none_allowed=True
    )


@dataclass
class GeneralizedHiddenMarkovModelInstanceConfig(InstanceConfig):
    """Configuration for the generalized hidden markov model."""

    transition_matrices: jax.Array
    initial_state: jax.Array | None = None

    def __init__(
        self,
        transition_matrices: jax.Array,
        initial_state: jax.Array | None = None,
        _target_: str = "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel",
    ) -> None:
        super().__init__(_target_=_target_)
        self.transition_matrices = transition_matrices
        self.initial_state = initial_state


def is_generalized_hidden_markov_model_target(target: str) -> bool:
    """Check if the target is a generalized hidden markov model target."""
    return target == "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel"


def is_generalized_hidden_markov_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generalized hidden markov model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generalized_hidden_markov_model_target(target)
    return False


def validate_generalized_hidden_markov_model_instance_config(cfg: DictConfig) -> None:
    """Validate a GeneralizedHiddenMarkovModelInstanceConfig.

    Args:
        cfg: A DictConfig with GeneralizedHiddenMarkovModelInstanceConfig fields (from Hydra).
    """
    transition_matrices = cfg.get("transition_matrices")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg,
        expected_target="simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel",
    )
    _validate_transition_matrices(transition_matrices, "GeneralizedHiddenMarkovModelInstanceConfig.transition_matrices")
    assert isinstance(transition_matrices, jax.Array)
    _validate_initial_state(
        initial_state, transition_matrices.shape[1], "GeneralizedHiddenMarkovModelInstanceConfig.initial_state"
    )


@dataclass
class HiddenMarkovModelInstanceConfig(InstanceConfig):
    """Configuration for the hidden markov model."""

    transition_matrices: jax.Array
    initial_state: jax.Array | None = None

    def __init__(
        self,
        transition_matrices: jax.Array,
        initial_state: jax.Array | None = None,
        _target_: str = "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel",
    ) -> None:
        super().__init__(_target_=_target_)
        self.transition_matrices = transition_matrices
        self.initial_state = initial_state


def is_hidden_markov_model_target(target: str) -> bool:
    """Check if the target is a hidden markov model target."""
    return target.startswith("simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel")


def is_hidden_markov_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a hidden markov model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_hidden_markov_model_target(target)
    return False


def validate_hidden_markov_model_instance_config(cfg: DictConfig) -> None:
    """Validate a HiddenMarkovModelInstanceConfig.

    Args:
        cfg: A DictConfig with HiddenMarkovModelInstanceConfig fields (from Hydra).
    """
    transition_matrices = cfg.get("transition_matrices")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel"
    )
    _validate_transition_matrices(transition_matrices, "HiddenMarkovModelInstanceConfig.transition_matrices")
    assert isinstance(transition_matrices, jax.Array)
    _validate_initial_state(
        initial_state, transition_matrices.shape[1], "HiddenMarkovModelInstanceConfig.initial_state"
    )


@dataclass
class GenerativeProcessConfig:
    """Base configuration for generative processes."""

    instance: InstanceConfig
    name: str | None = None
    base_vocab_size: int = MISSING
    bos_token: int | None = MISSING
    eos_token: int | None = MISSING
    vocab_size: int = MISSING


def is_generative_process_target(target: str) -> bool:
    """Check if the target is a generative process target."""
    return target.startswith("simplexity.generative_processes.")


def is_generative_process_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generative process config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generative_process_target(target)
    return False


def validate_generative_process_config(cfg: DictConfig) -> None:
    """Validate a GenerativeProcessConfig.

    Args:
        cfg: A DictConfig with instance, name, base_vocab_size, bos_token, eos_token,
             and vocab_size fields (from Hydra).
    """
    instance = cfg.get("instance")
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("GenerativeProcessConfig.instance is required")

    if is_generalized_hidden_markov_model_builder_config(instance):
        validate_generalized_hidden_markov_model_builder_instance_config(instance)
    elif is_hidden_markov_model_builder_config(instance):
        validate_hidden_markov_model_builder_instance_config(instance)
    else:
        _validate_instance_config(instance)
    _validate_nonempty_str(cfg.get("name"), "GenerativeProcessConfig.name", is_none_allowed=True)

    base_vocab_size = cfg.get("base_vocab_size")
    if OmegaConf.is_missing(cfg, "base_vocab_size"):
        SIMPLEXITY_LOGGER.debug("[generative process] base_vocab_size is missing, will be resolved dynamically")
    else:
        _validate_positive_int(base_vocab_size, "GenerativeProcessConfig.base_vocab_size")

    # Validate token values
    bos_token = cfg.get("bos_token")
    eos_token = cfg.get("eos_token")
    vocab_size = cfg.get("vocab_size")

    if not OmegaConf.is_missing(cfg, "vocab_size"):
        _validate_positive_int(vocab_size, "GenerativeProcessConfig.vocab_size")

    if OmegaConf.is_missing(cfg, "bos_token"):
        SIMPLEXITY_LOGGER.debug("[generative process] bos_token is missing, will be resolved dynamically")
    elif bos_token is not None:
        _validate_non_negative_int(bos_token, "GenerativeProcessConfig.bos_token", is_none_allowed=True)
        if not OmegaConf.is_missing(cfg, "vocab_size") and bos_token >= vocab_size:
            raise ConfigValidationError(
                f"GenerativeProcessConfig.bos_token ({bos_token}) must be < vocab_size ({vocab_size})"
            )

    if OmegaConf.is_missing(cfg, "eos_token"):
        SIMPLEXITY_LOGGER.debug("[generative process] eos_token is missing, will be resolved dynamically")
    elif eos_token is not None:
        _validate_non_negative_int(eos_token, "GenerativeProcessConfig.eos_token", is_none_allowed=True)
        if not OmegaConf.is_missing(cfg, "vocab_size") and eos_token >= vocab_size:
            raise ConfigValidationError(
                f"GenerativeProcessConfig.eos_token ({eos_token}) must be < vocab_size ({vocab_size})"
            )

    # Ensure tokens are distinct if both are set (skip if either is MISSING)
    if (
        not OmegaConf.is_missing(cfg, "bos_token")
        and not OmegaConf.is_missing(cfg, "eos_token")
        and bos_token is not None
        and eos_token is not None
        and bos_token == eos_token
    ):
        raise ConfigValidationError(f"GenerativeProcessConfig.bos_token and eos_token cannot be the same ({bos_token})")

    if OmegaConf.is_missing(cfg, "vocab_size"):
        SIMPLEXITY_LOGGER.debug("[generative process] vocab_size is missing, will be resolved dynamically")
    else:
        # Only validate consistency if base_vocab_size is also resolved
        if not OmegaConf.is_missing(cfg, "base_vocab_size"):
            _validate_positive_int(base_vocab_size, "GenerativeProcessConfig.base_vocab_size")
            use_bos_token = bos_token is not None or OmegaConf.is_missing(cfg, "bos_token")
            use_eos_token = eos_token is not None or OmegaConf.is_missing(cfg, "eos_token")
            expected_vocab_size = base_vocab_size + use_bos_token + use_eos_token
            if vocab_size != expected_vocab_size:
                raise ConfigValidationError(
                    f"GenerativeProcessConfig.vocab_size ({vocab_size}) must be equal to "
                    f"base_vocab_size ({base_vocab_size}) "
                    f"+ use_bos_token ({use_bos_token}) "
                    f"+ use_eos_token ({use_eos_token}) "
                    f"= {expected_vocab_size}"
                )


@dynamic_resolve
def resolve_generative_process_config(cfg: DictConfig, base_vocab_size: int) -> None:
    """Resolve the GenerativeProcessConfig."""
    # Resolve base_vocab_size
    if OmegaConf.is_missing(cfg, "base_vocab_size"):
        cfg.base_vocab_size = base_vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] base_vocab_size resolved to: %s", base_vocab_size)
    elif cfg.get("base_vocab_size") != base_vocab_size:
        raise ConfigValidationError(
            f"GenerativeProcessConfig.base_vocab_size ({cfg.get('base_vocab_size')}) must be equal to {base_vocab_size}"
        )
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] base_vocab_size defined as: %s", cfg.get("base_vocab_size"))
    vocab_size = base_vocab_size
    # Resolve bos_token
    if OmegaConf.is_missing(cfg, "bos_token"):
        cfg.bos_token = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] bos_token resolved to: %s", cfg.bos_token)
        vocab_size += 1
    elif cfg.get("bos_token", None) is not None:
        bos_token = cfg.get("bos_token")
        SIMPLEXITY_LOGGER.debug("[generative process] bos_token defined as: %s", bos_token)
        vocab_size = max(vocab_size, bos_token + 1)
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] no bos_token set")
    # Resolve eos_token
    if OmegaConf.is_missing(cfg, "eos_token"):
        cfg.eos_token = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] eos_token resolved to: %s", cfg.eos_token)
        vocab_size += 1
    elif cfg.get("eos_token", None) is not None:
        eos_token = cfg.get("eos_token")
        SIMPLEXITY_LOGGER.debug("[generative process] eos_token defined as: %s", eos_token)
        vocab_size = max(vocab_size, eos_token + 1)
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] no eos_token set")
    # Resolve vocab_size
    if OmegaConf.is_missing(cfg, "vocab_size"):
        cfg.vocab_size = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] vocab_size resolved to: %s", vocab_size)
    elif cfg.get("vocab_size") != vocab_size:
        raise ConfigValidationError(
            f"GenerativeProcessConfig.vocab_size ({cfg.get('vocab_size')}) must be equal to {vocab_size}"
        )
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] vocab_size defined as: %s", cfg.get("vocab_size"))

