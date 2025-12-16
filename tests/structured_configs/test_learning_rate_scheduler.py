"""Tests for learning rate scheduler configuration validation."""

import pytest
from omegaconf import OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.learning_rate_scheduler import (
    is_lr_scheduler_config,
    is_reduce_lr_on_plateau_config,
    is_windowed_reduce_lr_on_plateau_config,
    validate_lr_scheduler_config,
    validate_reduce_lr_on_plateau_instance_config,
    validate_windowed_reduce_lr_on_plateau_instance_config,
)


class TestIsReduceLROnPlateauConfig:
    """Tests for is_reduce_lr_on_plateau_config."""

    def test_is_reduce_lr_on_plateau_config(self):
        """Test that ReduceLROnPlateau target is correctly identified."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau"})
        assert is_reduce_lr_on_plateau_config(cfg) is True

    def test_is_reduce_lr_on_plateau_config_wrong_target(self):
        """Test that non-ReduceLROnPlateau target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_reduce_lr_on_plateau_config(cfg) is False

    def test_is_reduce_lr_on_plateau_config_no_target(self):
        """Test that missing _target_ returns False."""
        cfg = OmegaConf.create({})
        assert is_reduce_lr_on_plateau_config(cfg) is False


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


class TestIsLrSchedulerConfig:
    """Tests for is_lr_scheduler_config."""

    def test_is_lr_scheduler_config_reduce_on_plateau(self):
        """Test is_lr_scheduler_config with ReduceLROnPlateau target."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau"})
        assert is_lr_scheduler_config(cfg) is True

    def test_is_lr_scheduler_config_windowed(self):
        """Test is_lr_scheduler_config with WindowedReduceLROnPlateau target."""
        cfg = OmegaConf.create({"_target_": "simplexity.lr_schedulers.WindowedReduceLROnPlateau"})
        assert is_lr_scheduler_config(cfg) is True

    def test_is_lr_scheduler_config_other_scheduler(self):
        """Test is_lr_scheduler_config with other scheduler target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR"})
        assert is_lr_scheduler_config(cfg) is False

    def test_is_lr_scheduler_config_optimizer(self):
        """Test is_lr_scheduler_config with optimizer target returns False."""
        cfg = OmegaConf.create({"_target_": "torch.optim.Adam"})
        assert is_lr_scheduler_config(cfg) is False

    def test_is_lr_scheduler_config_no_target(self):
        """Test is_lr_scheduler_config with missing _target_."""
        cfg = OmegaConf.create({})
        assert is_lr_scheduler_config(cfg) is False


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


class TestValidateLrSchedulerConfig:
    """Tests for validate_lr_scheduler_config."""

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

    def test_valid_with_name(self):
        """Test validation passes with optional name field."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                },
                "name": "my_scheduler",
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

    def test_invalid_not_plateau_scheduler(self):
        """Test validation fails when target is not a plateau scheduler."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                },
            }
        )
        with pytest.raises(ConfigValidationError, match="must be ReduceLROnPlateau or WindowedReduceLROnPlateau"):
            validate_lr_scheduler_config(cfg)

    def test_invalid_optimizer_target(self):
        """Test validation fails when target is an optimizer."""
        cfg = OmegaConf.create(
            {
                "instance": {
                    "_target_": "torch.optim.Adam",
                },
            }
        )
        with pytest.raises(ConfigValidationError, match="must be ReduceLROnPlateau or WindowedReduceLROnPlateau"):
            validate_lr_scheduler_config(cfg)
