"""muP (Maximal Update Parameterization) configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.validation import validate_bool, validate_positive_int


@dataclass
class MuPConfig:
    """Configuration for muP (Maximal Update Parameterization).

    Attributes:
        enabled: Whether to apply muP to the predictive model.
        base_d_model: Width of the base shape model used for muP shape inference.
        base_d_head: Head dimension of the base model. Defaults to `base_d_model // n_heads`.
        base_d_mlp: MLP hidden dimension of the base model. Defaults to `4 * base_d_model`.
        delta_d_model: Width of the delta shape model. Must differ from `base_d_model`.
        delta_d_head: Head dimension of the delta model. Defaults to `delta_d_model // n_heads`.
        delta_d_mlp: MLP hidden dimension of the delta model. Defaults to `4 * delta_d_model`.
        rescale_params: Whether `mup.set_base_shapes` should rescale initialized weights.
    """

    enabled: bool = False
    base_d_model: int = 64
    base_d_head: int | None = None
    base_d_mlp: int | None = None
    delta_d_model: int = 128
    delta_d_head: int | None = None
    delta_d_mlp: int | None = None
    rescale_params: bool = True


def validate_mup_config(cfg: DictConfig) -> None:
    """Validate a MuPConfig.

    Args:
        cfg: A DictConfig with MuPConfig fields (from Hydra).
    """
    enabled = cfg.get("enabled")
    base_d_model = cfg.get("base_d_model")
    base_d_head = cfg.get("base_d_head")
    base_d_mlp = cfg.get("base_d_mlp")
    delta_d_model = cfg.get("delta_d_model")
    delta_d_head = cfg.get("delta_d_head")
    delta_d_mlp = cfg.get("delta_d_mlp")
    rescale_params = cfg.get("rescale_params")

    validate_bool(enabled, "MuPConfig.enabled", is_none_allowed=True)
    validate_positive_int(base_d_model, "MuPConfig.base_d_model")
    validate_positive_int(base_d_head, "MuPConfig.base_d_head", is_none_allowed=True)
    validate_positive_int(base_d_mlp, "MuPConfig.base_d_mlp", is_none_allowed=True)
    validate_positive_int(delta_d_model, "MuPConfig.delta_d_model")
    validate_positive_int(delta_d_head, "MuPConfig.delta_d_head", is_none_allowed=True)
    validate_positive_int(delta_d_mlp, "MuPConfig.delta_d_mlp", is_none_allowed=True)
    validate_bool(rescale_params, "MuPConfig.rescale_params", is_none_allowed=True)

    if base_d_model == delta_d_model:
        raise ConfigValidationError(
            f"MuPConfig.base_d_model ({base_d_model}) must differ from MuPConfig.delta_d_model ({delta_d_model})"
        )
