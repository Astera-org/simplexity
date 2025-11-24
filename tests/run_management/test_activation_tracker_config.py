"""Tests for activation tracker structured config validation and instantiation."""

import jax.numpy as jnp
import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management.run_management import (
    _instantiate_activation_tracker,
    _setup_activation_trackers,
)
from simplexity.run_management.structured_configs import (
    ConfigValidationError,
    is_activation_analysis_target,
    is_activation_tracker_target,
    validate_activation_analysis_config,
    validate_activation_tracker_config,
)


@pytest.fixture
def tracker_cfg() -> DictConfig:
    """Provides a valid activation tracker config for testing."""
    return OmegaConf.create(
        {
            "activation_tracker": {
                "instance": {
                    "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                    "analyses": {
                        "pca": {
                            "name": "pca_custom",
                            "instance": {
                                "_target_": "simplexity.activations.activation_analyses.PcaAnalysis",
                                "n_components": 1,
                                "last_token_only": True,
                                "concat_layers": False,
                                "use_probs_as_weights": True,
                            },
                        },
                        "linear": {
                            "instance": {
                                "_target_": "simplexity.activations.activation_analyses.LinearRegressionAnalysis",
                                "fit_intercept": True,
                                "last_token_only": False,
                                "concat_layers": True,
                                "use_probs_as_weights": False,
                            }
                        },
                    },
                }
            }
        }
    )


def test_validate_activation_tracker_config_accepts_instance_wrapped_analyses(tracker_cfg: DictConfig) -> None:
    """Tests that a valid activation tracker config passes validation."""
    validate_activation_tracker_config(tracker_cfg.activation_tracker)


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


def test_activation_analysis_target_helpers() -> None:
    """Activation analysis helpers should flag valid prefixes and tracker targets."""
    assert is_activation_analysis_target("simplexity.activations.foo.Bar")
    assert not is_activation_analysis_target("other.module")
    assert is_activation_tracker_target("simplexity.activations.activation_tracker.ActivationTracker")
    assert not is_activation_tracker_target("simplexity.activations.other")


def test_validate_activation_analysis_config_errors() -> None:
    """Validator should enforce instance presence and target namespace."""
    cfg_missing_instance = OmegaConf.create({"name": "bad"})
    with pytest.raises(ConfigValidationError, match="instance is required"):
        validate_activation_analysis_config(cfg_missing_instance)

    cfg_bad_target = OmegaConf.create(
        {
            "instance": {"_target_": "other.module"},
            "name": "foo",
        }
    )
    with pytest.raises(ConfigValidationError, match="must be an activation analysis target"):
        validate_activation_analysis_config(cfg_bad_target)

    cfg_empty_name = OmegaConf.create(
        {
            "instance": {"_target_": "simplexity.activations.some.Analysis"},
            "name": "",
        }
    )
    with pytest.raises(ConfigValidationError, match="name"):
        validate_activation_analysis_config(cfg_empty_name)


def test_validate_activation_tracker_config_errors() -> None:
    """Tracker validator should reject invalid nested analyses."""
    cfg_missing_analyses = OmegaConf.create(
        {
            "instance": {
                "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
            }
        }
    )
    with pytest.raises(ConfigValidationError, match="analyses is required"):
        validate_activation_tracker_config(cfg_missing_analyses)

    cfg_not_dict = OmegaConf.create(
        {
            "instance": {
                "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                "analyses": 5,
            }
        }
    )
    with pytest.raises(ConfigValidationError, match="must be a dictionary"):
        validate_activation_tracker_config(cfg_not_dict)

    cfg_missing_instance = OmegaConf.create(
        {
            "instance": {
                "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                "analyses": {"pca": {}},
            }
        }
    )
    with pytest.raises(ConfigValidationError, match="must specify an InstanceConfig"):
        validate_activation_tracker_config(cfg_missing_instance)

    cfg_bad_target = OmegaConf.create(
        {
            "instance": {
                "_target_": "simplexity.activations.activation_tracker.ActivationTracker",
                "analyses": {
                    "pca": {
                        "instance": {
                            "_target_": "other.module",
                        }
                    }
                },
            }
        }
    )
    with pytest.raises(ConfigValidationError, match="must target an activation analysis"):
        validate_activation_tracker_config(cfg_bad_target)


def test_setup_activation_trackers_integration(monkeypatch: pytest.MonkeyPatch, tracker_cfg: DictConfig) -> None:
    """Setup helper should instantiate trackers for filtered keys."""
    captured_instance_keys = []

    def fake_filter(cfg_arg, instance_keys, *_, **__):
        assert cfg_arg is tracker_cfg
        captured_instance_keys.extend(instance_keys)
        return ["activation_tracker.instance"]

    monkeypatch.setattr(
        "simplexity.run_management.run_management.filter_instance_keys",
        fake_filter,
    )

    instantiated = {}

    def fake_instantiate(_, key):
        instantiated[key] = object()
        return instantiated[key]

    monkeypatch.setattr(
        "simplexity.run_management.run_management._instantiate_activation_tracker",
        fake_instantiate,
    )

    trackers = _setup_activation_trackers(tracker_cfg, ["activation_tracker.instance"])
    assert trackers == {"activation_tracker.instance": instantiated["activation_tracker.instance"]}


def test_setup_activation_trackers_no_instances_logs(monkeypatch: pytest.MonkeyPatch, tracker_cfg: DictConfig) -> None:
    """When no instance keys survive filtering the helper returns None."""

    def fake_filter(*_, **__):
        return []

    monkeypatch.setattr(
        "simplexity.run_management.run_management.filter_instance_keys",
        fake_filter,
    )

    trackers = _setup_activation_trackers(tracker_cfg, ["activation_tracker.instance"])
    assert trackers is None


def test_instantiate_activation_tracker_builds_analysis_objects(tracker_cfg: DictConfig) -> None:
    """Tests that the activation tracker and its analyses are instantiated correctly."""
    tracker = _instantiate_activation_tracker(tracker_cfg, "activation_tracker.instance")
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
