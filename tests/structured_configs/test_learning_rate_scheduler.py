"""Tests for learning rate scheduler configuration validation."""

import pytest
from omegaconf import OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.learning_rate_scheduler import (
    is_cosine_annealing_lr_config,
    is_cosine_annealing_warm_restarts_config,
    is_exponential_lr_config,
    is_linear_lr_config,
    is_lr_scheduler_config,
    is_lr_scheduler_target,
    is_reduce_lr_on_plateau_config,
    is_step_lr_config,
    is_windowed_reduce_lr_on_plateau_config,
    validate_cosine_annealing_lr_instance_config,
    validate_cosine_annealing_warm_restarts_instance_config,
    validate_exponential_lr_instance_config,
    validate_linear_lr_instance_config,
    validate_lr_scheduler_config,
    validate_reduce_lr_on_plateau_instance_config,
    validate_step_lr_instance_config,
    validate_windowed_reduce_lr_on_plateau_instance_config,
)


class TestIsSchedulerConfig:
    """Tests for is_*_config functions."""

    def test_is_step_lr_config(self):
        """Test that StepLR target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_step_lr_config(cfg) is True

    def test_is_step_lr_config_wrong_target(self):
        """Test that non-StepLR target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ExponentialLR"})
        assert is_step_lr_config(cfg) is False

    def test_is_step_lr_config_no_target(self):
        """Test that missing _target_ returns False."""
        cfg = OmegaConf.create({})
        assert is_step_lr_config(cfg) is False

    def test_is_exponential_lr_config(self):
        """Test that ExponentialLR target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ExponentialLR"})
        assert is_exponential_lr_config(cfg) is True

    def test_is_exponential_lr_config_wrong_target(self):
        """Test that non-ExponentialLR target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_exponential_lr_config(cfg) is False

    def test_is_cosine_annealing_lr_config(self):
        """Test that CosineAnnealingLR target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR"})
        assert is_cosine_annealing_lr_config(cfg) is True

    def test_is_cosine_annealing_lr_config_wrong_target(self):
        """Test that non-CosineAnnealingLR target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_cosine_annealing_lr_config(cfg) is False

    def test_is_cosine_annealing_warm_restarts_config(self):
        """Test that CosineAnnealingWarmRestarts target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"})
        assert is_cosine_annealing_warm_restarts_config(cfg) is True

    def test_is_cosine_annealing_warm_restarts_config_wrong_target(self):
        """Test that non-CosineAnnealingWarmRestarts target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_cosine_annealing_warm_restarts_config(cfg) is False

    def test_is_linear_lr_config(self):
        """Test that LinearLR target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.LinearLR"})
        assert is_linear_lr_config(cfg) is True

    def test_is_linear_lr_config_wrong_target(self):
        """Test that non-LinearLR target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_linear_lr_config(cfg) is False

    def test_is_reduce_lr_on_plateau_config(self):
        """Test that ReduceLROnPlateau target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau"})
        assert is_reduce_lr_on_plateau_config(cfg) is True

    def test_is_reduce_lr_on_plateau_config_wrong_target(self):
        """Test that non-ReduceLROnPlateau target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_reduce_lr_on_plateau_config(cfg) is False


class TestIsLrSchedulerTarget:
    """Tests for is_lr_scheduler_target and is_lr_scheduler_config."""

    def test_is_lr_scheduler_target_true(self):
        """Test that lr_scheduler targets are correctly identified."""
        assert is_lr_scheduler_target("torch.optim.lr_scheduler.StepLR") is True
        assert is_lr_scheduler_target("torch.optim.lr_scheduler.ExponentialLR") is True

    def test_is_lr_scheduler_target_false(self):
        """Test that non-scheduler targets return False."""
        assert is_lr_scheduler_target("torch.optim.Adam") is False
        assert is_lr_scheduler_target("some.other.Module") is False

    def test_is_lr_scheduler_config(self):
        """Test is_lr_scheduler_config with valid scheduler target."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_lr_scheduler_config(cfg) is True

    def test_is_lr_scheduler_config_false(self):
        """Test is_lr_scheduler_config with optimizer target."""
        cfg = OmegaConf.create({"_target_": "torch.optim.Adam"})
        assert is_lr_scheduler_config(cfg) is False

    def test_is_lr_scheduler_config_no_target(self):
        """Test is_lr_scheduler_config with missing _target_."""
        cfg = OmegaConf.create({})
        assert is_lr_scheduler_config(cfg) is False


class TestValidateStepLR:
    """Tests for validate_step_lr_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid StepLR config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
                "gamma": 0.1,
                "last_epoch": -1,
            }
        )
        validate_step_lr_instance_config(cfg)

    def test_invalid_step_size(self):
        """Test validation fails with negative step_size."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": -1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_step_lr_instance_config(cfg)

    def test_invalid_gamma(self):
        """Test validation fails with negative gamma."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "gamma": -0.1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_step_lr_instance_config(cfg)

    def test_invalid_last_epoch_type(self):
        """Test validation fails with non-int last_epoch."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "last_epoch": "invalid",
            }
        )
        with pytest.raises(ConfigValidationError, match="last_epoch must be an int"):
            validate_step_lr_instance_config(cfg)


class TestValidateExponentialLR:
    """Tests for validate_exponential_lr_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid ExponentialLR config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                "gamma": 0.95,
                "last_epoch": -1,
            }
        )
        validate_exponential_lr_instance_config(cfg)

    def test_invalid_gamma(self):
        """Test validation fails with negative gamma."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                "gamma": -0.5,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_exponential_lr_instance_config(cfg)

    def test_invalid_last_epoch_type(self):
        """Test validation fails with non-int last_epoch."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                "last_epoch": 1.5,
            }
        )
        with pytest.raises(ConfigValidationError, match="last_epoch must be an int"):
            validate_exponential_lr_instance_config(cfg)


class TestValidateCosineAnnealingLR:
    """Tests for validate_cosine_annealing_lr_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid CosineAnnealingLR config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 100,
                "eta_min": 0.0,
                "last_epoch": -1,
            }
        )
        validate_cosine_annealing_lr_instance_config(cfg)

    def test_invalid_t_max(self):
        """Test validation fails with zero T_max."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_cosine_annealing_lr_instance_config(cfg)

    def test_invalid_eta_min(self):
        """Test validation fails with negative eta_min."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "eta_min": -0.1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_cosine_annealing_lr_instance_config(cfg)

    def test_invalid_last_epoch_type(self):
        """Test validation fails with non-int last_epoch."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "last_epoch": "bad",
            }
        )
        with pytest.raises(ConfigValidationError, match="last_epoch must be an int"):
            validate_cosine_annealing_lr_instance_config(cfg)


class TestValidateCosineAnnealingWarmRestarts:
    """Tests for validate_cosine_annealing_warm_restarts_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid CosineAnnealingWarmRestarts config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                "T_0": 10,
                "T_mult": 2,
                "eta_min": 0.001,
                "last_epoch": -1,
            }
        )
        validate_cosine_annealing_warm_restarts_instance_config(cfg)

    def test_invalid_t_0(self):
        """Test validation fails with zero T_0."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                "T_0": 0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_cosine_annealing_warm_restarts_instance_config(cfg)

    def test_invalid_t_mult(self):
        """Test validation fails with negative T_mult."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                "T_mult": -1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_cosine_annealing_warm_restarts_instance_config(cfg)

    def test_invalid_last_epoch_type(self):
        """Test validation fails with non-int last_epoch."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                "last_epoch": [],
            }
        )
        with pytest.raises(ConfigValidationError, match="last_epoch must be an int"):
            validate_cosine_annealing_warm_restarts_instance_config(cfg)


class TestValidateLinearLR:
    """Tests for validate_linear_lr_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid LinearLR config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": 0.333,
                "end_factor": 1.0,
                "total_iters": 5,
                "last_epoch": -1,
            }
        )
        validate_linear_lr_instance_config(cfg)

    def test_invalid_start_factor(self):
        """Test validation fails with zero start_factor."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": 0.0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_linear_lr_instance_config(cfg)

    def test_invalid_end_factor(self):
        """Test validation fails with negative end_factor."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "end_factor": -1.0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_linear_lr_instance_config(cfg)

    def test_invalid_total_iters(self):
        """Test validation fails with zero total_iters."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "total_iters": 0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_linear_lr_instance_config(cfg)

    def test_invalid_last_epoch_type(self):
        """Test validation fails with non-int last_epoch."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "last_epoch": {},
            }
        )
        with pytest.raises(ConfigValidationError, match="last_epoch must be an int"):
            validate_linear_lr_instance_config(cfg)


class TestValidateReduceLROnPlateau:
    """Tests for validate_reduce_lr_on_plateau_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid ReduceLROnPlateau config."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.1,
                "patience": 10,
                "threshold": 1e-4,
                "cooldown": 0,
                "min_lr": 0.0,
                "eps": 1e-8,
            }
        )
        validate_reduce_lr_on_plateau_instance_config(cfg)

    def test_valid_max_mode(self):
        """Test validation passes with mode='max'."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "mode": "max",
            }
        )
        validate_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "mode": "invalid",
            }
        )
        with pytest.raises(ConfigValidationError, match="mode must be 'min' or 'max'"):
            validate_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_factor(self):
        """Test validation fails with zero factor."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "factor": 0.0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_patience(self):
        """Test validation fails with negative patience."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "patience": -1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_cooldown(self):
        """Test validation fails with negative cooldown."""
        cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "cooldown": -5,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_reduce_lr_on_plateau_instance_config(cfg)


class TestValidateLrSchedulerConfig:
    """Tests for validate_lr_scheduler_config."""

    def test_valid_step_lr(self):
        """Test validation passes with valid StepLR scheduler config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 10,
                },
                "name": "my_scheduler",
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_exponential_lr(self):
        """Test validation passes with valid ExponentialLR scheduler config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.ExponentialLR",
                    "gamma": 0.9,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_cosine_annealing_lr(self):
        """Test validation passes with valid CosineAnnealingLR scheduler config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": 50,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_cosine_annealing_warm_restarts(self):
        """Test validation passes with valid CosineAnnealingWarmRestarts config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                    "T_0": 10,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_linear_lr(self):
        """Test validation passes with valid LinearLR scheduler config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.LinearLR",
                    "total_iters": 10,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_reduce_lr_on_plateau(self):
        """Test validation passes with valid ReduceLROnPlateau config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                    "patience": 5,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_valid_unknown_scheduler(self):
        """Test validation passes with unknown but valid scheduler target."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.OneCycleLR",
                    "max_lr": 0.1,
                },
            }
        )
        validate_lr_scheduler_config(cfg)

    def test_invalid_instance_not_dict(self):
        """Test validation fails when instance is not a DictConfig."""
        cfg = OmegaConf.create(
            {
                "instance": "not_a_dict",
            }
        )
        with pytest.raises(ConfigValidationError, match="instance must be a DictConfig"):
            validate_lr_scheduler_config(cfg)

    def test_invalid_not_scheduler_target(self):
        """Test validation fails when target is not a scheduler."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.Adam",
                },
            }
        )
        with pytest.raises(ConfigValidationError, match="must be a learning rate scheduler target"):
            validate_lr_scheduler_config(cfg)

    def test_valid_windowed_reduce_lr_on_plateau(self):
        """Test validation passes with valid WindowedReduceLROnPlateau config."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                    "window_size": 10,
                    "update_every": 100,
                    "patience": 5,
                },
            }
        )
        validate_lr_scheduler_config(cfg)


class TestIsWindowedReduceLROnPlateauConfig:
    """Tests for is_windowed_reduce_lr_on_plateau_config."""

    def test_is_windowed_reduce_lr_on_plateau_config(self):
        """Test that WindowedReduceLROnPlateau target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau"})
        assert is_windowed_reduce_lr_on_plateau_config(cfg) is True

    def test_is_windowed_reduce_lr_on_plateau_config_wrong_target(self):
        """Test that non-WindowedReduceLROnPlateau target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau"})
        assert is_windowed_reduce_lr_on_plateau_config(cfg) is False

    def test_is_windowed_reduce_lr_on_plateau_config_no_target(self):
        """Test that missing _target_ returns False."""
        cfg = OmegaConf.create({})
        assert is_windowed_reduce_lr_on_plateau_config(cfg) is False


class TestValidateWindowedReduceLROnPlateau:
    """Tests for validate_windowed_reduce_lr_on_plateau_instance_config."""

    def test_valid_config(self):
        """Test validation passes with valid WindowedReduceLROnPlateau config."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "window_size": 10,
                "update_every": 100,
                "mode": "min",
                "factor": 0.1,
                "patience": 10,
                "threshold": 1e-4,
                "cooldown": 0,
                "min_lr": 0.0,
                "eps": 1e-8,
            }
        )
        validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_valid_max_mode(self):
        """Test validation passes with mode='max'."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "mode": "max",
            }
        )
        validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "mode": "invalid",
            }
        )
        with pytest.raises(ConfigValidationError, match="mode must be 'min' or 'max'"):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_window_size(self):
        """Test validation fails with zero window_size."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "window_size": 0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_update_every(self):
        """Test validation fails with zero update_every."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "update_every": 0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_factor(self):
        """Test validation fails with zero factor."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "factor": 0.0,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_patience(self):
        """Test validation fails with negative patience."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "patience": -1,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)

    def test_invalid_cooldown(self):
        """Test validation fails with negative cooldown."""
        cfg = OmegaConf.create(
            {
                "_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
                "cooldown": -5,
            }
        )
        with pytest.raises(ConfigValidationError):
            validate_windowed_reduce_lr_on_plateau_instance_config(cfg)
