"""Tests for predictive-model structured configs."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import re
from unittest.mock import call, patch

import pytest
from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError, DeviceResolutionError
from simplexity.structured_configs.predictive_model import (
    HookedTransformerConfigConfig,
    HookedTransformerInstancecConfig,
    PredictiveModelConfig,
    is_hooked_transformer_config,
    is_predictive_model_config,
    is_predictive_model_target,
    resolve_hooked_transformer_config,
    validate_hooked_transformer_config,
    validate_hooked_transformer_config_config,
    validate_predictive_model_config,
)


<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
class TestHookedTransformerConfig:
    """Test HookedTransformerConfig."""

    """Test PredictiveModelConfig."""
=======
class TestHookedTransformerConfig:  # pylint: disable=too-many-public-methods
    """Test HookedTransformerConfig."""
>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py

    def test_hooked_transformer_config(self) -> None:
        """Test creating hooked transformer config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(
            PredictiveModelConfig(
                instance=HookedTransformerInstancecConfig(
                    cfg=HookedTransformerConfigConfig(
                        n_layers=2,
                        d_model=128,
                        d_head=32,
                        n_ctx=16,
                        d_vocab=3,
                    ),
                )
            )
        )
        assert OmegaConf.select(cfg, "instance._target_") == "transformer_lens.HookedTransformer"
        assert OmegaConf.select(cfg, "instance.cfg._target_") == "transformer_lens.HookedTransformerConfig"
        assert OmegaConf.select(cfg, "instance.cfg.n_layers") == 2
        assert OmegaConf.select(cfg, "instance.cfg.d_model") == 128
        assert OmegaConf.select(cfg, "instance.cfg.d_head") == 32
        assert OmegaConf.select(cfg, "instance.cfg.n_ctx") == 16
        assert OmegaConf.select(cfg, "instance.cfg.n_heads") == -1
        assert OmegaConf.select(cfg, "instance.cfg.d_mlp") is None
        assert OmegaConf.select(cfg, "instance.cfg.d_vocab") == 3
        assert OmegaConf.select(cfg, "instance.cfg.act_fn") is None
        assert OmegaConf.select(cfg, "instance.cfg.normalization_type") == "LN"
        assert OmegaConf.select(cfg, "instance.cfg.device") is None
        assert OmegaConf.select(cfg, "instance.cfg.seed") is None
        assert cfg.get("name") is None
        assert cfg.get("load_checkpoint_step") is None

    def test_validate_hooked_transformer_config_config_valid(self) -> None:
        """Test validate_hooked_transformer_config_config with valid configs."""
        # Test with minimal config (n_heads defaults to -1)
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": MISSING,
            }
        )
        validate_hooked_transformer_config_config(cfg)

        # Test with full config including n_heads
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_mlp": 512,
                "act_fn": None,
                "d_vocab": 1000,
                "normalization_type": None,
                "device": None,
                "seed": 42,
            }
        )
        validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_incompatible_dimensions(self) -> None:
        """Test validate_hooked_transformer_config_config raises when n_heads=-1.

        Specifically tests that d_model must be divisible by d_head when n_heads=-1.
        """
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 130,  # Not divisible by d_head (130 % 32 == 2)
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_mlp": 512,
                "act_fn": None,
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": None,
                "seed": None,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_model.*must be divisible by d_head"):
            validate_hooked_transformer_config_config(cfg)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("d_model", 128),
            ("d_head", 32),
            ("n_heads", -1),
            ("n_layers", 2),
            ("d_mlp", 512),
            ("d_vocab", 1000),
            ("n_ctx", 256),
        ],
    )
    def test_validate_hooked_transformer_config_config_invalid_fields(self, field, value) -> None:
        """Test validate_hooked_transformer_config_config raises for invalid field values."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )

        # Non-integer value
        cfg[field] = str(value)
        with pytest.raises(ConfigValidationError, match=f"HookedTransformerConfigConfig.{field} must be an int"):
            validate_hooked_transformer_config_config(cfg)

        cfg[field] = False
        with pytest.raises(ConfigValidationError, match=f"HookedTransformerConfigConfig.{field} must be positive"):
            validate_hooked_transformer_config_config(cfg)

        # Non-positive value (0 should fail for all fields, including n_heads)
        cfg[field] = 0
        with pytest.raises(ConfigValidationError, match=f"HookedTransformerConfigConfig.{field} must be positive"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_d_model_not_divisible_by_n_heads(self) -> None:
        """Test validate_hooked_transformer_config_config raises when d_model is not divisible by n_heads."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 130,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_model.*must be divisible by n_heads"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_d_head_times_n_heads_not_equal_d_model(self) -> None:
        """Test validate_hooked_transformer_config_config raises when d_head * n_heads != d_model."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 30,
                "n_ctx": 256,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_head.*n_heads.*must equal d_model"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_valid(self) -> None:
        """Test validate_hooked_transformer_config with valid configs."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformer",
                "cfg": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformerConfig",
                        "n_layers": 2,
                        "d_model": 128,
                        "d_head": 32,
                        "n_ctx": 256,
                        "n_heads": 4,
                        "d_mlp": 512,
                        "act_fn": "relu",
                        "d_vocab": MISSING,
                        "normalization_type": "LN",
                        "device": "cpu",
                        "seed": 42,
                    }
                ),
            }
        )
        validate_hooked_transformer_config(cfg)

    def test_validate_hooked_transformer_config_missing_target(self) -> None:
        """Test validate_hooked_transformer_config raises when _target_ is missing."""
        cfg = DictConfig(
            {
                "cfg": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformerConfig",
                        "n_layers": 2,
                        "d_model": 128,
                        "d_head": 32,
                        "n_ctx": 256,
                        "n_heads": 4,
                        "d_mlp": 512,
                        "act_fn": "relu",
                        "d_vocab": MISSING,
                        "normalization_type": "LN",
                        "device": "cpu",
                        "seed": 42,
                    }
                ),
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_hooked_transformer_config(cfg)

    def test_validate_hooked_transformer_config_missing_cfg(self) -> None:
        """Test validate_hooked_transformer_config raises when cfg is missing."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        with pytest.raises(ConfigValidationError, match="HookedTransformerConfig.cfg is required"):
            validate_hooked_transformer_config(cfg)

    def test_resolve_hooked_transformer_config_without_kwargs(self) -> None:
        """Test resolve_hooked_transformer_config with valid configs."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": MISSING,
            }
        )
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.debug") as mock_debug,
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.info") as mock_info,
            patch("simplexity.structured_configs.predictive_model.resolve_device", return_value="cpu"),
        ):
            resolve_hooked_transformer_config(cfg)
            mock_debug.assert_called_once_with("[predictive model] no vocab_size set")
            mock_info.assert_called_once_with("[predictive model] device resolved to: %s", "cpu")
        assert OmegaConf.is_missing(cfg, "d_vocab")
        assert cfg.get("device") == "cpu"

    def test_resolve_hooked_transformer_config_with_complete_values(self) -> None:
<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
=======
        """Test resolve_hooked_transformer_config with complete values."""
>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": 4,
                "device": "cuda",
            }
        )
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.debug") as mock_debug,
            patch("simplexity.structured_configs.predictive_model.resolve_device", return_value="cuda"),
        ):
            resolve_hooked_transformer_config(cfg, vocab_size=4)
            mock_debug.assert_has_calls(
                [
                    call("[predictive model] d_vocab defined as: %s", 4),
                    call("[predictive model] device defined as: %s", "cuda"),
                ]
            )
        assert cfg.get("d_vocab") == 4
        assert cfg.get("device") == "cuda"

    def test_resolve_hooked_transformer_config_with_missing_values(self) -> None:
<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
=======
        """Test resolve_hooked_transformer_config with missing values."""
>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": MISSING,
                "device": None,
            }
        )
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.info") as mock_info,
            patch("simplexity.structured_configs.predictive_model.resolve_device", return_value="cuda"),
        ):
            resolve_hooked_transformer_config(cfg, vocab_size=4)
            mock_info.assert_has_calls(
                [
                    call("[predictive model] d_vocab resolved to: %s", 4),
                    call("[predictive model] device resolved to: %s", "cuda"),
                ]
            )
        assert cfg.get("d_vocab") == 4
        assert cfg.get("device") == "cuda"

    def test_resolve_hooked_transformer_config_with_invalid_values(self) -> None:
<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
=======
        """Test resolve_hooked_transformer_config with invalid values."""
>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": 3,
            }
        )
        with pytest.raises(
            ConfigValidationError, match=re.escape("HookedTransformerConfig.d_vocab (3) must be equal to 4")
        ):
            resolve_hooked_transformer_config(cfg, vocab_size=4)

    def test_resolve_hooked_transformer_config_with_conflicting_device(self) -> None:
        """Test resolve_hooked_transformer_config with conflicting device."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
                "n_ctx": 8,
                "n_heads": -1,
                "d_vocab": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError, match=re.escape("HookedTransformerConfig.n_ctx (8) must be equal to 16")
        ):
            resolve_hooked_transformer_config(cfg, bos_token=3, sequence_len=16)

    def test_resolve_hooked_transformer_config_with_conflicting_device(self) -> None:
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": MISSING,
=======
                "n_ctx": 256,
>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py
                "n_heads": -1,
                "d_vocab": MISSING,
                "device": "cuda",
            }
        )
        error = DeviceResolutionError("boom")
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.warning") as mock_warning,
            patch("simplexity.structured_configs.predictive_model.resolve_device", side_effect=error),
        ):
            resolve_hooked_transformer_config(cfg)
            mock_warning.assert_has_calls(
                [
                    call(
                        "[predictive model] specified device %s could not be resolved: %s",
                        "cuda",
                        error,
                    ),
                    call("[predictive model] specified device %s resolved to %s", "cuda", "cpu"),
                ]
            )
        assert cfg.get("device") == "cpu"

<<<<<<< HEAD:tests/run_management/test_predictive_model_config.py
=======
    def test_resolve_hooked_transformer_config_device_mismatch_updates_cfg(self) -> None:
        """Test resolve_hooked_transformer_config device mismatch updates config."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_vocab": MISSING,
                "device": "cuda",
            }
        )
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.warning") as mock_warning,
            patch("simplexity.structured_configs.predictive_model.resolve_device", return_value="cpu"),
        ):
            resolve_hooked_transformer_config(cfg)
            mock_warning.assert_called_once_with("[predictive model] specified device %s resolved to %s", "cuda", "cpu")
        assert cfg.get("device") == "cpu"

    def test_resolve_hooked_transformer_config_device_auto(self) -> None:
        """Test resolve_hooked_transformer_config with auto device."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 64,
                "d_head": 32,
                "n_ctx": 128,
                "n_heads": -1,
                "d_vocab": MISSING,
                "device": "auto",
            }
        )
        with (
            patch("simplexity.structured_configs.predictive_model.SIMPLEXITY_LOGGER.info") as mock_info,
            patch("simplexity.structured_configs.predictive_model.resolve_device", return_value="mps"),
        ):
            resolve_hooked_transformer_config(cfg)
            mock_info.assert_any_call("[predictive model] device resolved to: %s", "mps")
        assert cfg.get("device") == "mps"

>>>>>>> origin/main:tests/structured_configs/test_predictive_model_config.py
    def test_is_predictive_model_target_valid(self) -> None:
        """Test is_predictive_model_target with valid model targets."""
        assert is_predictive_model_target("transformer_lens.HookedTransformer")
        assert is_predictive_model_target("torch.nn.Linear")
        assert is_predictive_model_target("equinox.nn.Linear")
        assert is_predictive_model_target("penzai.nn.Linear")
        assert is_predictive_model_target("penzai.models.Transformer")
        assert is_predictive_model_target("simplexity.predictive_models.MyModel")

    def test_is_predictive_model_target_invalid(self) -> None:
        """Test is_predictive_model_target with invalid targets."""
        assert not is_predictive_model_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_predictive_model_target("torch.optim.Adam")
        assert not is_predictive_model_target("")

    def test_is_predictive_model_config_valid(self) -> None:
        """Test is_predictive_model_config with valid predictive model configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        assert is_predictive_model_config(cfg)

        cfg = DictConfig({"_target_": "torch.nn.Linear"})
        assert is_predictive_model_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.predictive_models.MyModel"})
        assert is_predictive_model_config(cfg)

    def test_is_predictive_model_config_invalid(self) -> None:
        """Test is_predictive_model_config with invalid configs."""
        # Non-model target
        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_predictive_model_config(cfg)

        # Missing _target_
        cfg = DictConfig({"other_field": "value"})
        assert not is_predictive_model_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_predictive_model_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_predictive_model_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_predictive_model_config(cfg)

    def test_is_hooked_transformer_config_valid(self) -> None:
        """Test is_hooked_transformer_config with valid HookedTransformer configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        assert is_hooked_transformer_config(cfg)

    def test_is_hooked_transformer_config_invalid(self) -> None:
        """Test is_hooked_transformer_config with invalid configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformerConfig"})
        assert not is_hooked_transformer_config(cfg)

        cfg = DictConfig({"_target_": "torch.nn.Linear"})
        assert not is_hooked_transformer_config(cfg)

        cfg = DictConfig({})
        assert not is_hooked_transformer_config(cfg)

    def test_validate_predictive_model_config_valid(self) -> None:
        """Test validate_predictive_model_config with valid configs."""
        # Valid config without name or load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformer",
                        "cfg": DictConfig(
                            {
                                "_target_": "transformer_lens.HookedTransformerConfig",
                                "n_layers": 2,
                                "d_model": 128,
                                "d_head": 32,
                                "n_ctx": 256,
                                "n_heads": 4,
                                "d_mlp": 512,
                                "act_fn": "relu",
                                "d_vocab": MISSING,
                                "normalization_type": "LN",
                                "device": "cpu",
                                "seed": 42,
                            }
                        ),
                    }
                ),
            }
        )
        validate_predictive_model_config(cfg)

        # Valid config with name and load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "my_model",
                "load_checkpoint_step": 100,
            }
        )
        validate_predictive_model_config(cfg)

        # Valid config with None name and load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": None,
                "load_checkpoint_step": None,
            }
        )
        validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_missing_instance(self) -> None:
        """Test validate_predictive_model_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.instance is required"):
            validate_predictive_model_config(cfg)

        cfg = DictConfig({"name": "my_model"})
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.instance is required"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_instance(self) -> None:
        """Test validate_predictive_model_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_predictive_model_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_predictive_model_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_non_model_target(self) -> None:
        """Test validate_predictive_model_config raises when instance target is not a model target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.instance must be a predictive model target"
        ):
            validate_predictive_model_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.instance must be a predictive model target"
        ):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_name(self) -> None:
        """Test validate_predictive_model_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a non-empty string"):
            validate_predictive_model_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a non-empty string"):
            validate_predictive_model_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a string or None"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_load_checkpoint_step(self) -> None:
        """Test validate_predictive_model_config raises when load_checkpoint_step is invalid."""
        # Non-integer load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "load_checkpoint_step": "100",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.load_checkpoint_step must be an int or None"
        ):
            validate_predictive_model_config(cfg)

        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "load_checkpoint_step": False,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.load_checkpoint_step must be an int or None"
        ):
            validate_predictive_model_config(cfg)

        # Negative load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "load_checkpoint_step": -1,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.load_checkpoint_step must be non-negative"
        ):
            validate_predictive_model_config(cfg)
