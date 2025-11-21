"""Tests for activation tracker structured config validation and instantiation."""

import jax.numpy as jnp
import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management.run_management import _instantiate_activation_tracker
from simplexity.run_management.structured_configs import ConfigValidationError, validate_activation_tracker_config


def _build_valid_tracker_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "activation_tracker": {
                "instance": {
                    "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                    "analyses": {
                        "pca": {
                            "name": "pca_custom",
                            "instance": {
                                "_target_": "simplexity.activations.activation_analyses.PCAAnalysis",
                                "n_components": 1,
                            },
                        },
                        "linear": {
                            "instance": {
                                "_target_": "simplexity.activations.activation_analyses.LinearRegressionAnalysis",
                            }
                        },
                    },
                }
            }
        }
    )


def test_validate_activation_tracker_config_accepts_instance_wrapped_analyses() -> None:
    """Tests that a valid activation tracker config passes validation."""
    cfg = _build_valid_tracker_cfg()
    validate_activation_tracker_config(cfg.activation_tracker)


def test_validate_activation_tracker_config_requires_instance_block() -> None:
    """Tests that missing 'instance' block raises ConfigValidationError."""
    bad_cfg = OmegaConf.create(
        {
            "instance": {
                "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                "analyses": {
                    "pca": {
                        "_target_": "simplexity.activations.activation_analyses.PCAAnalysis",
                    }
                },
            }
        }
    )
    with pytest.raises(ConfigValidationError):
        validate_activation_tracker_config(bad_cfg)


def test_instantiate_activation_tracker_builds_analysis_objects() -> None:
    """Tests that the activation tracker and its analyses are instantiated correctly."""
    cfg = _build_valid_tracker_cfg()
    tracker = _instantiate_activation_tracker(cfg, "activation_tracker.instance")
    assert tracker is not None

    inputs = jnp.array([[0, 1]], dtype=jnp.int32)
    beliefs = jnp.ones((1, 2, 2), dtype=jnp.float32) * 0.5
    probs = jnp.ones((1, 2), dtype=jnp.float32) * 0.5
    activations = {"layer": jnp.ones((1, 2, 4), dtype=jnp.float32)}

    scalars, projections = tracker.analyze(
        inputs=inputs,
        beliefs=beliefs,
        probs=probs,
        activations=activations,
    )
    assert "pca_custom/layer_cumvar_1" in scalars
    assert any(key.startswith("linear/") for key in projections)
