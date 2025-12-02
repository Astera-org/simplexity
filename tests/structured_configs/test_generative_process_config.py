"""Tests for GenerativeProcessConfig validation.

This module contains tests for generative process configuration validation, including
validation of generative process targets, vocab sizes, special tokens (BOS/EOS),
and generative process configuration instances.
"""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import re
from typing import Any
from unittest.mock import call, patch

import jax.numpy as jnp
import pytest
from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.generative_process import (
    GeneralizedHiddenMarkovModelBuilderInstanceConfig,
    GeneralizedHiddenMarkovModelInstanceConfig,
    GenerativeProcessConfig,
    HiddenMarkovModelBuilderInstanceConfig,
    HiddenMarkovModelInstanceConfig,
    InstanceConfig,
    NonergodicHiddenMarkovModelBuilderInstanceConfig,
    is_generalized_hidden_markov_model_builder_config,
    is_generalized_hidden_markov_model_builder_target,
    is_generalized_hidden_markov_model_config,
    is_generalized_hidden_markov_model_target,
    is_generative_process_config,
    is_generative_process_target,
    is_hidden_markov_model_builder_config,
    is_hidden_markov_model_builder_target,
    is_hidden_markov_model_config,
    is_hidden_markov_model_target,
    is_nonergodic_hidden_markov_model_builder_config,
    is_nonergodic_hidden_markov_model_builder_target,
    resolve_generative_process_config,
    validate_generalized_hidden_markov_model_builder_instance_config,
    validate_generalized_hidden_markov_model_instance_config,
    validate_generative_process_config,
    validate_hidden_markov_model_builder_instance_config,
    validate_hidden_markov_model_instance_config,
    validate_nonergodic_hidden_markov_model_builder_instance_config,
)


def _with_missing_tokens(cfg: DictConfig) -> DictConfig:
    """Ensure bos/eos tokens are treated as missing."""
    cfg["bos_token"] = MISSING
    cfg["eos_token"] = MISSING
    return cfg


def _builder_instance(**overrides: Any) -> DictConfig:
    data: dict[str, Any] = {
        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
        "process_name": "mess3",
        "process_params": DictConfig({}),
    }
    data.update(overrides)
    return DictConfig(data)


class TestGenerativeProcessBuilders:
    """Tests covering the specialized builder configs."""

    def test_hidden_markov_model_builder_detection_and_validation(self) -> None:
        target = "simplexity.generative_processes.builder.build_hidden_markov_model"
        cfg = DictConfig(
            {
                "_target_": target,
                "process_name": "mess3",
                "process_params": DictConfig({"alpha": 0.1}),
                "initial_state": [0.6, 0.4],
            }
        )
        assert is_hidden_markov_model_builder_target(target)
        assert is_hidden_markov_model_builder_config(cfg)
        validate_hidden_markov_model_builder_instance_config(cfg)
        structured = HiddenMarkovModelBuilderInstanceConfig(
            process_name="mess3",
            process_params={},
            initial_state=[0.5, 0.5],
        )
        assert structured.process_params == {}

        cfg.process_name = ""
        with pytest.raises(ConfigValidationError, match="process_name must be a non-empty string"):
            validate_hidden_markov_model_builder_instance_config(cfg)

    def test_generalized_hidden_markov_model_builder_validation(self) -> None:
        target = "simplexity.generative_processes.builder.build_generalized_hidden_markov_model"
        cfg = DictConfig(
            {
                "_target_": target,
                "process_name": "mess3",
                "process_params": DictConfig({"alpha": 0.1}),
                "initial_state": [0.5, 0.5],
            }
        )
        assert is_generalized_hidden_markov_model_builder_target(target)
        assert is_generalized_hidden_markov_model_builder_config(cfg)
        validate_generalized_hidden_markov_model_builder_instance_config(cfg)
        structured = GeneralizedHiddenMarkovModelBuilderInstanceConfig(
            process_name="mess3",
            process_params={},
            initial_state=[0.25, 0.75],
        )
        assert structured.initial_state == [0.25, 0.75]

        cfg.initial_state = 123
        with pytest.raises(ConfigValidationError, match="initial_state must be a sequence"):
            validate_generalized_hidden_markov_model_builder_instance_config(cfg)

    def test_nonergodic_hidden_markov_model_builder_validation(self) -> None:
        target = "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model"
        cfg = DictConfig(
            {
                "_target_": target,
                "process_names": ["a", "b"],
                "process_params": [DictConfig({"alpha": 0.1}), DictConfig({"beta": 0.2})],
                "process_weights": [0.5, 0.5],
                "vocab_maps": [[0, 1], [1, 2]],
                "add_bos_token": True,
            }
        )
        assert is_nonergodic_hidden_markov_model_builder_target(target)
        assert is_nonergodic_hidden_markov_model_builder_config(cfg)
        validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)
        structured = NonergodicHiddenMarkovModelBuilderInstanceConfig(
            process_names=["a"],
            process_params=[DictConfig({"alpha": 0.1})],
            process_weights=[1.0],
        )
        assert structured.process_weights == [1.0]

        cfg.process_weights = [0.5]
        with pytest.raises(ConfigValidationError, match="must have the same length"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

        cfg.process_weights = [0.5, 0.5]
        cfg.add_bos_token = "yes"
        with pytest.raises(ConfigValidationError, match="add_bos_token must be a bool"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

    def test_nonergodic_hidden_markov_model_builder_sequence_types(self) -> None:
        base = {
            "_target_": "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model",
            "process_names": ["a", "b"],
            "process_params": [DictConfig({}), DictConfig({})],
            "process_weights": [0.5, 0.5],
        }
        cfg = DictConfig({**base, "process_names": 123})
        with pytest.raises(ConfigValidationError, match="process_names must be a list"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

        cfg = DictConfig({**base, "process_params": 123})
        with pytest.raises(ConfigValidationError, match="process_params must be a sequence"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

        cfg = DictConfig({**base, "process_weights": 123})
        with pytest.raises(ConfigValidationError, match="process_weights must be a sequence"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

        cfg = DictConfig({**base, "vocab_maps": 123})
        with pytest.raises(ConfigValidationError, match="vocab_maps must be a sequence"):
            validate_nonergodic_hidden_markov_model_builder_instance_config(cfg)

    def test_hidden_markov_model_instance_validation(self) -> None:
        transition_matrices = jnp.ones((2, 2, 2), dtype=jnp.float32)
        initial_state = jnp.ones((2,), dtype=jnp.float32)
        cfg = DictConfig(
            {
                "_target_": "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel",
                "transition_matrices": transition_matrices,
                "initial_state": initial_state,
            },
            flags={"allow_objects": True},
        )
        assert is_hidden_markov_model_target(cfg._target_)
        assert is_hidden_markov_model_config(cfg)
        validate_hidden_markov_model_instance_config(cfg)
        structured_instance = HiddenMarkovModelInstanceConfig(
            transition_matrices=transition_matrices,
            initial_state=initial_state,
        )
        assert structured_instance.transition_matrices.shape[0] == 2

        cfg.initial_state = jnp.ones((3,), dtype=jnp.float32)
        with pytest.raises(ConfigValidationError, match="must have the same number of elements"):
            validate_hidden_markov_model_instance_config(cfg)

    def test_generalized_hidden_markov_model_instance_validation(self) -> None:
        transition_matrices = jnp.ones((2, 2, 2), dtype=jnp.float32)
        initial_state = jnp.ones((2,), dtype=jnp.float32)
        target = "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel"
        cfg = DictConfig(
            {
                "_target_": target,
                "transition_matrices": transition_matrices,
                "initial_state": initial_state,
            },
            flags={"allow_objects": True},
        )
        assert is_generalized_hidden_markov_model_target(cfg._target_)
        assert is_generalized_hidden_markov_model_config(cfg)
        validate_generalized_hidden_markov_model_instance_config(cfg)
        structured_instance = GeneralizedHiddenMarkovModelInstanceConfig(
            transition_matrices=transition_matrices,
            initial_state=initial_state,
        )
        assert structured_instance.initial_state is not None
        assert structured_instance.initial_state.shape[0] == 2

        cfg.initial_state = jnp.ones((1,), dtype=jnp.float32)
        with pytest.raises(ConfigValidationError, match="must have the same number of elements"):
            validate_generalized_hidden_markov_model_instance_config(cfg)


class TestGenerativeProcessConfig:
    """Test GenerativeProcessConfig."""

    def test_generative_process_config(self) -> None:
        """Test creating generative process config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(
            GenerativeProcessConfig(
                instance=InstanceConfig(_target_="some_target"),
                base_vocab_size=3,
                bos_token=None,
                eos_token=None,
                vocab_size=3,
            )
        )
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None
        assert cfg.get("base_vocab_size") == 3
        assert cfg.get("bos_token") is None
        assert cfg.get("eos_token") is None
        assert cfg.get("vocab_size") == 3

<<<<<<< HEAD:tests/run_management/test_generative_process_config.py
=======
    def test_validate_generative_process_config_handles_generalized_builder(self) -> None:
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_generalized_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({}),
                        "initial_state": [0.5, 0.5],
                    }
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        validate_generative_process_config(cfg)

    def test_validate_generative_process_config_handles_nonergodic_builder(self) -> None:
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model",
                        "process_names": ["a"],
                        "process_params": [DictConfig({})],
                        "process_weights": [1.0],
                    }
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        validate_generative_process_config(cfg)

    def test_validate_generative_process_config_handles_generalized_instance(self) -> None:
        transition_matrices = jnp.ones((2, 2, 2), dtype=jnp.float32)
        initial_state = jnp.ones((2,), dtype=jnp.float32)
        target = "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel"
        instance_cfg = DictConfig(
            {
                "_target_": target,
                "transition_matrices": transition_matrices,
                "initial_state": initial_state,
            },
            flags={"allow_objects": True},
        )
        cfg = DictConfig(
            {
                "instance": instance_cfg,
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        validate_generative_process_config(cfg)

    def test_validate_generative_process_config_handles_hidden_instance(self) -> None:
        transition_matrices = jnp.ones((2, 2, 2), dtype=jnp.float32)
        initial_state = jnp.ones((2,), dtype=jnp.float32)
        instance_cfg = DictConfig(
            {
                "_target_": "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel",
                "transition_matrices": transition_matrices,
                "initial_state": initial_state,
            },
            flags={"allow_objects": True},
        )
        cfg = DictConfig(
            {
                "instance": instance_cfg,
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        validate_generative_process_config(cfg)

>>>>>>> origin/main:tests/structured_configs/test_generative_process_config.py
    def test_is_generative_process_target_valid(self) -> None:
        """Test is_generative_process_target with valid generative process targets."""
        assert is_generative_process_target("simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel")
        assert is_generative_process_target("simplexity.generative_processes.builder.build_hidden_markov_model")

    def test_is_generative_process_target_invalid(self) -> None:
        """Test is_generative_process_target with invalid targets."""
        assert not is_generative_process_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert not is_generative_process_target("torch.optim.Adam")
        assert not is_generative_process_target("")

    def test_is_generative_process_config_valid(self) -> None:
        """Test is_generative_process_config with valid generative process configs."""
        cfg = DictConfig({"_target_": "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel"})
        assert is_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                "process_name": "mess3",
                "process_params": DictConfig({"x": 0.15, "a": 0.6}),
            }
        )
        assert is_generative_process_config(cfg)

    def test_is_generative_process_config_invalid(self) -> None:
        """Test is_generative_process_config with invalid configs."""
        # Non-generative process target
        cfg = DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})
        assert not is_generative_process_config(cfg)

        # Missing _target_
        cfg = DictConfig({"process_name": "mess3", "process_params": DictConfig({"x": 0.15, "a": 0.6})})
        assert not is_generative_process_config(cfg)

        # _target_ is not a omegaconf target
        cfg = DictConfig({"target": "simplexity.generative_processes.builder.build_hidden_markov_model"})
        assert not is_generative_process_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_generative_process_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_generative_process_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_generative_process_config(cfg)

    def test_validate_generative_process_config_valid(self) -> None:
        """Test validate_generative_process_config with valid configs."""
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({"x": 0.15, "a": 0.6}),
                    }
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({"x": 0.15, "a": 0.6}),
                    }
                ),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": 5,
            }
        )
        validate_generative_process_config(cfg)

    def test_validate_generative_process_config_missing_instance(self) -> None:
        """Test validate_generative_process_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.instance is required"):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                "process_name": "mess3",
                "process_params": DictConfig({"x": 0.15, "a": 0.6}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.instance is required"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_instance(self) -> None:
        """Test validate_generative_process_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig(
            {
                "instance": DictConfig({"process_name": "mess3", "process_params": DictConfig({"x": 0.15, "a": 0.6})}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_generative_process_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""}), "base_vocab_size": MISSING, "vocab_size": MISSING})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_generative_process_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123}), "base_vocab_size": MISSING, "vocab_size": MISSING})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_non_generative_process_target(self) -> None:
        """Test validate_generative_process_config raises when instance target is not a generative process target."""
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="GenerativeProcessConfig.instance must be a generative process target"
        ):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.optim.Adam"}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="GenerativeProcessConfig.instance must be a generative process target",
        ):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_name(self) -> None:
        """Test validate_generative_process_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({}),
                    }
                ),
                "name": "",
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a non-empty string"):
            validate_generative_process_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({}),
                    }
                ),
                "name": "   ",
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a non-empty string"):
            validate_generative_process_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                        "process_name": "mess3",
                        "process_params": DictConfig({}),
                    }
                ),
                "name": 123,
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a string or None"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_base_vocab_size(self) -> None:
        """Test validate_generative_process_config raises when base_vocab_size is invalid."""
        # Non-integer base_vocab_size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": "3",
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be an int"):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": False,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be positive"):
            validate_generative_process_config(cfg)

        # Negative base_vocab_size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": -1,
                "vocab_size": MISSING,
            }
        )
        cfg = _with_missing_tokens(cfg)
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be positive"):
            validate_generative_process_config(cfg)

    @pytest.mark.parametrize("token_type", ["bos_token", "eos_token"])
    def test_validate_generative_process_config_invalid_special_tokens(self, token_type: str) -> None:
        """Test validate_generative_process_config raises when special tokens are invalid."""
        other_token = "eos_token" if token_type == "bos_token" else "bos_token"
        # Non-integer token value
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: "3",
                "vocab_size": MISSING,
            }
        )
        cfg[other_token] = MISSING
        with pytest.raises(
            ConfigValidationError,
            match=re.escape(f"GenerativeProcessConfig.{token_type} must be an int or None, got <class 'str'>"),
        ):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: False,
                "vocab_size": MISSING,
            }
        )
        cfg[other_token] = MISSING
        with pytest.raises(ConfigValidationError, match=f"GenerativeProcessConfig.{token_type} must be an int or None"):
            validate_generative_process_config(cfg)

        # Negative token value
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: -1,
                "vocab_size": MISSING,
            }
        )
        cfg[other_token] = MISSING
        with pytest.raises(ConfigValidationError, match=f"GenerativeProcessConfig.{token_type} must be non-negative"):
            validate_generative_process_config(cfg)

        # Token value greater than vocab size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: 4,
                "vocab_size": 4,
            }
        )
        cfg[other_token] = MISSING
        with pytest.raises(
            ConfigValidationError, match=re.escape(f"GenerativeProcessConfig.{token_type} (4) must be < vocab_size (4)")
        ):
            validate_generative_process_config(cfg)

    @pytest.mark.parametrize(
        ("attribute", "is_vocab_token"),
        [
            ("base_vocab_size", False),
            ("bos_token", True),
            ("eos_token", True),
            ("vocab_size", False),
        ],
    )
    def test_validate_generative_process_config_missing_attribute(self, attribute: str, is_vocab_token: bool):
        """Test validate_generative_process_config raises when an attribute is missing."""
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": 5,
            }
        )
        cfg[attribute] = MISSING
        # assert that there is a SIMPLEXITY_LOGGER debug log
        with patch("simplexity.structured_configs.generative_process.SIMPLEXITY_LOGGER.debug") as mock_debug:
            validate_generative_process_config(cfg)
            mock_debug.assert_any_call(f"[generative process] {attribute} is missing, will be resolved dynamically")

    def test_validate_generative_process_config_invalid_bos_eos_token_same_value(self):
        """Test validate_generative_process_config raises when bos_token and eos_token are the same."""
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 3,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="GenerativeProcessConfig.bos_token and eos_token cannot be the same"
        ):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_vocab_size(self):
        """Test validate_generative_process_config raises when vocab_size is invalid."""
        # Non-integer vocab size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": "4",
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.vocab_size must be an int"):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": False,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.vocab_size must be positive"):
            validate_generative_process_config(cfg)

        # Negative vocab size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": -1,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.vocab_size must be positive"):
            validate_generative_process_config(cfg)

        # Incorrect vocab size
        cfg = DictConfig(
            {
                "instance": _builder_instance(),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": 6,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match=re.escape(
                "GenerativeProcessConfig.vocab_size (6) must be equal to "
                "base_vocab_size (3) + use_bos_token (True) + use_eos_token (True) = 5"
            ),
        ):
            validate_generative_process_config(cfg)

    def test_resolve_generative_process_config_with_complete_values(self):
        """Test _resolve_generative_process_config correctly calculates vocab_size when tokens are explicitly set."""
        cfg = DictConfig(
            {
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": 4,
                "vocab_size": 5,
            }
        )
        with patch("simplexity.structured_configs.generative_process.SIMPLEXITY_LOGGER.debug") as mock_debug:
            resolve_generative_process_config(cfg, base_vocab_size=3)
            mock_debug.assert_has_calls(
                [
                    call("[generative process] base_vocab_size defined as: %s", 3),
                    call("[generative process] bos_token defined as: %s", 3),
                    call("[generative process] eos_token defined as: %s", 4),
                    call("[generative process] vocab_size defined as: %s", 5),
                ]
            )
        assert cfg.get("base_vocab_size") == 3
        assert cfg.get("bos_token") == 3
        assert cfg.get("eos_token") == 4
        assert cfg.get("vocab_size") == 5

    def test_resolve_generative_process_config_with_none_values(self):
        cfg = DictConfig(
            {
                "base_vocab_size": 3,
                "bos_token": None,
                "eos_token": None,
                "vocab_size": 3,
            }
        )
        with patch("simplexity.structured_configs.generative_process.SIMPLEXITY_LOGGER.debug") as mock_debug:
            resolve_generative_process_config(cfg, base_vocab_size=3)
            mock_debug.assert_has_calls(
                [
                    call("[generative process] base_vocab_size defined as: %s", 3),
                    call("[generative process] no bos_token set"),
                    call("[generative process] no eos_token set"),
                    call("[generative process] vocab_size defined as: %s", 3),
                ]
            )
            assert mock_debug.call_count == 4
        assert cfg.get("base_vocab_size") == 3
        assert cfg.get("bos_token") is None
        assert cfg.get("eos_token") is None
        assert cfg.get("vocab_size") == 3

    def test_resolve_generative_process_config_with_missing_values(self):
        """Test _resolve_generative_process_config correctly calculates vocab_size when tokens are explicitly set."""
        cfg = DictConfig(
            {
                "base_vocab_size": MISSING,
                "bos_token": MISSING,
                "eos_token": MISSING,
                "vocab_size": MISSING,
            }
        )
        with patch("simplexity.structured_configs.generative_process.SIMPLEXITY_LOGGER.info") as mock_info:
            resolve_generative_process_config(cfg, base_vocab_size=3)
            mock_info.assert_has_calls(
                [
                    call("[generative process] base_vocab_size resolved to: %s", 3),
                    call("[generative process] bos_token resolved to: %s", 3),
                    call("[generative process] eos_token resolved to: %s", 4),
                    call("[generative process] vocab_size resolved to: %s", 5),
                ]
            )
            assert mock_info.call_count == 4
        assert cfg.get("base_vocab_size") == 3
        assert cfg.get("bos_token") == 3
        assert cfg.get("eos_token") == 4
        assert cfg.get("vocab_size") == 5

    def test_resolve_generative_process_config_with_invalid_values(self):
        """Test _resolve_generative_process_config raises when values are invalid."""
        cfg = DictConfig(
            {
                "base_vocab_size": 4,
                "bos_token": None,
                "eos_token": None,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match=re.escape("GenerativeProcessConfig.base_vocab_size (4) must be equal to 3"),
        ):
            resolve_generative_process_config(cfg, base_vocab_size=3)

        cfg = DictConfig(
            {
                "base_vocab_size": 3,
                "bos_token": None,
                "eos_token": None,
                "vocab_size": 4,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match=re.escape("GenerativeProcessConfig.vocab_size (4) must be equal to 3"),
        ):
            resolve_generative_process_config(cfg, base_vocab_size=3)
