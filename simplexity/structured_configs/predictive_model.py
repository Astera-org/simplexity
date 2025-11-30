"""Predictive model configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError, DeviceResolutionError
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import (
    validate_non_negative_int,
    validate_nonempty_str,
    validate_positive_int,
)
from simplexity.utils.config_utils import dynamic_resolve
from simplexity.utils.pytorch_utils import resolve_device


@dataclass
class HookedTransformerConfigConfig(InstanceConfig):
    """Configuration for HookedTransformerConfig."""

    n_layers: int
    d_model: int
    d_head: int
    n_ctx: int
    n_heads: int = -1
    d_mlp: int | None = None
    act_fn: str | None = None
    d_vocab: int = MISSING
    normalization_type: str | None = "LN"
    device: str | None = None
    seed: int | None = None

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        n_ctx: int,
        n_heads: int = -1,
        d_mlp: int | None = None,
        act_fn: str | None = None,
        d_vocab: int = MISSING,
        normalization_type: str | None = "LN",
        device: str | None = None,
        seed: int | None = None,
        _target_: str = "transformer_lens.HookedTransformerConfig",
    ):
        super().__init__(_target_=_target_)
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.act_fn = act_fn
        self.d_vocab = d_vocab
        self.normalization_type = normalization_type
        self.device = device
        self.seed = seed


def validate_hooked_transformer_config_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerConfigConfig.

    Args:
        cfg: A DictConfig with HookedTransformerConfigConfig fields (from Hydra).
    """
    n_layers = cfg.get("n_layers")
    d_model = cfg.get("d_model")
    d_head = cfg.get("d_head")
    n_ctx = cfg.get("n_ctx")
    n_heads = cfg.get("n_heads")
    d_mlp = cfg.get("d_mlp")
    act_fn = cfg.get("act_fn")
    normalization_type = cfg.get("normalization_type")
    device = cfg.get("device")
    seed = cfg.get("seed")

    validate_instance_config(cfg, expected_target="transformer_lens.HookedTransformerConfig")
    validate_positive_int(n_layers, "HookedTransformerConfigConfig.n_layers")
    validate_positive_int(d_model, "HookedTransformerConfigConfig.d_model")
    validate_positive_int(d_head, "HookedTransformerConfigConfig.d_head")
    validate_positive_int(n_ctx, "HookedTransformerConfigConfig.n_ctx")
    if n_heads != -1:
        validate_positive_int(n_heads, "HookedTransformerConfigConfig.n_heads")
        if d_model % n_heads != 0:
            raise ConfigValidationError(
                f"HookedTransformerConfigConfig.d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if d_head * n_heads != d_model:
            raise ConfigValidationError(
                f"HookedTransformerConfigConfig.d_head ({d_head}) * n_heads ({n_heads}) must equal d_model ({d_model})"
            )
    elif d_model % d_head != 0:
        raise ConfigValidationError(
            f"HookedTransformerConfigConfig.d_model ({d_model}) must be divisible by d_head ({d_head})"
        )

    validate_positive_int(d_mlp, "HookedTransformerConfigConfig.d_mlp", is_none_allowed=True)
    validate_nonempty_str(act_fn, "HookedTransformerConfigConfig.act_fn", is_none_allowed=True)
    if OmegaConf.is_missing(cfg, "d_vocab"):
        SIMPLEXITY_LOGGER.debug("[predictive model] d_vocab is missing, will be resolved dynamically")
    else:
        d_vocab = cfg.get("d_vocab")
        validate_positive_int(d_vocab, "HookedTransformerConfigConfig.d_vocab")
    validate_nonempty_str(normalization_type, "HookedTransformerConfigConfig.normalization_type", is_none_allowed=True)
    validate_nonempty_str(device, "HookedTransformerConfigConfig.device", is_none_allowed=True)
    validate_non_negative_int(seed, "HookedTransformerConfigConfig.seed", is_none_allowed=True)


@dataclass
class HookedTransformerInstancecConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig

    def __init__(self, cfg: HookedTransformerConfigConfig, _target_: str = "transformer_lens.HookedTransformer"):
        super().__init__(_target_=_target_)
        self.cfg = cfg


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"


def validate_hooked_transformer_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerInstancecConfig.

    Args:
        cfg: A DictConfig with _target_ and cfg fields (from Hydra).
    """
    validate_instance_config(cfg)
    nested_cfg = cfg.get("cfg")
    if nested_cfg is None:
        raise ConfigValidationError("HookedTransformerConfig.cfg is required")
    validate_hooked_transformer_config_config(nested_cfg)


@dynamic_resolve
def resolve_hooked_transformer_config(cfg: DictConfig, *, vocab_size: int | None = None) -> None:
    """Resolve the HookedTransformerConfig."""
    # Resolve d_vocab
    if vocab_size is None:
        SIMPLEXITY_LOGGER.debug("[predictive model] no vocab_size set")
    else:
        if OmegaConf.is_missing(cfg, "d_vocab"):
            cfg.d_vocab = vocab_size
            SIMPLEXITY_LOGGER.info("[predictive model] d_vocab resolved to: %s", vocab_size)
        elif cfg.get("d_vocab") != vocab_size:
            raise ConfigValidationError(
                f"HookedTransformerConfig.d_vocab ({cfg.get('d_vocab')}) must be equal to {vocab_size}"
            )
        else:
            SIMPLEXITY_LOGGER.debug("[predictive model] d_vocab defined as: %s", cfg.get("d_vocab"))

    # Resolve device
    device: str | None = cfg.get("device", None)
    try:
        resolved_device = resolve_device(device)
    except DeviceResolutionError as e:
        SIMPLEXITY_LOGGER.warning("[predictive model] specified device %s could not be resolved: %s", device, e)
        resolved_device = "cpu"
    if device is None or device == "auto":
        cfg.device = resolved_device
        SIMPLEXITY_LOGGER.info("[predictive model] device resolved to: %s", resolved_device)
    elif device != resolved_device:
        cfg.device = resolved_device
        SIMPLEXITY_LOGGER.warning("[predictive model] specified device %s resolved to %s", device, resolved_device)
    else:
        SIMPLEXITY_LOGGER.debug("[predictive model] device defined as: %s", device)


@dataclass
class PredictiveModelConfig:
    """Base configuration for predictive models."""

    instance: InstanceConfig
    name: str | None = None
    load_checkpoint_step: int | None = None


def is_predictive_model_target(target: str) -> bool:
    """Check if the target is a predictive model target."""
    parts = target.split(".")
    if len(parts) > 2:
        if parts[1] == "nn":  # torch.nn, equinox.nn, penzai.nn
            return True
        if "models" in parts[1]:  # penzai.models
            return True
    return parts[0] == "transformer_lens"


def is_predictive_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_predictive_model_target(target)
    return False


def validate_predictive_model_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with instance, optional name, and optional load_checkpoint_step fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("PredictiveModelConfig.instance is required")
    name = cfg.get("name")
    load_checkpoint_step = cfg.get("load_checkpoint_step")

    if is_hooked_transformer_config(instance):
        validate_hooked_transformer_config(instance)
    else:
        validate_instance_config(instance)
        if not is_predictive_model_config(instance):
            raise ConfigValidationError("PredictiveModelConfig.instance must be a predictive model target")
    validate_nonempty_str(name, "PredictiveModelConfig.name", is_none_allowed=True)
    validate_non_negative_int(load_checkpoint_step, "PredictiveModelConfig.load_checkpoint_step", is_none_allowed=True)
