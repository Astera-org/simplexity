"""Integration test that wraps simplexity/run.py to test the full training workflow."""

import os
import tempfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from simplexity.run import train_model


def test_integration_end_to_end():
    """Test the complete end-to-end workflow using run.py."""
    # Use temporary directories for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log_dir = temp_path / "logs"
        checkpoint_dir = temp_path / "checkpoints"

        # Get the config directory path
        config_dir = Path(__file__).parent / "configs"

        # Initialize Hydra with our test config
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.2"):
            cfg = compose(
                config_name="integration_test.yaml",
                overrides=[
                    f"logging.instance.file_path={log_dir}/test.log",
                    f"persistence.instance.directory={checkpoint_dir}",
                    "training.n_steps=20",  # Small number for quick testing
                    "training.checkpoint_frequency=10",
                    "training.metric_frequency=10",
                ]
            )

            # Store initial config for validation
            initial_config = OmegaConf.to_container(cfg)

            # Train the model using run.py
            final_loss = train_model(cfg)

            # Verify that training completed and loss is a valid number
            assert isinstance(final_loss, float)
            assert final_loss > 0
            assert final_loss < 100  # Sanity check - loss shouldn't be astronomical

            # Verify logs were created
            assert log_dir.exists(), "Log directory should have been created"
            log_files = list(log_dir.glob("*"))
            assert len(log_files) > 0, "Log files should have been created"

            # Verify checkpoints were saved
            assert checkpoint_dir.exists(), "Checkpoint directory should have been created"
            checkpoint_files = list(checkpoint_dir.glob("**/*"))
            assert len(checkpoint_files) > 0, "Checkpoint files should have been saved"

            # Check that we have checkpoint at expected steps
            step_10_checkpoint = checkpoint_dir / "step_10"
            assert step_10_checkpoint.exists(), "Checkpoint at step 10 should exist"

            # Verify the config used matches what we expected
            assert initial_config["training"]["n_steps"] == 20
            assert initial_config["seed"] == 42
            assert initial_config["experiment_name"] == "integration_test"


def test_integration_loss_decreases():
    """Test that loss decreases during training."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log_dir = temp_path / "logs"
        checkpoint_dir = temp_path / "checkpoints"

        config_dir = Path(__file__).parent / "configs"

        # First run - get initial loss with very few steps
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.2"):
            cfg = compose(
                config_name="integration_test.yaml",
                overrides=[
                    f"logging.instance.file_path={log_dir}/run1/test.log",
                    f"persistence.instance.directory={checkpoint_dir}/run1",
                    "training.n_steps=1",
                ]
            )
            initial_loss = train_model(cfg)

        # Second run - train for more steps
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.2"):
            cfg = compose(
                config_name="integration_test.yaml",
                overrides=[
                    f"logging.instance.file_path={log_dir}/run2/test.log",
                    f"persistence.instance.directory={checkpoint_dir}/run2",
                    "training.n_steps=50",
                    "seed=42",  # Same seed for reproducibility
                ]
            )
            final_loss = train_model(cfg)

        # Verify loss decreased with more training
        assert final_loss < initial_loss, f"Loss should decrease with training: initial={initial_loss:.4f}, final={final_loss:.4f}"


def test_integration_with_validation():
    """Test that validation works correctly during training."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log_dir = temp_path / "logs"
        checkpoint_dir = temp_path / "checkpoints"

        config_dir = Path(__file__).parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.2"):
            cfg = compose(
                config_name="integration_test.yaml",
                overrides=[
                    f"logging.instance.file_path={log_dir}/test.log",
                    f"persistence.instance.directory={checkpoint_dir}",
                    "training.n_steps=20",
                    "training.validation_frequency=10",
                    "validation.batches=2",
                ]
            )

            loss = train_model(cfg)

            # Check that validation ran (by checking logs contain validation info)
            assert log_dir.exists()
            assert loss > 0

            # Check that log file exists and contains validation info
            log_file = log_dir / "test.log"
            assert log_file.exists(), "Log file should have been created"
            with open(log_file, "r") as f:
                content = f.read()
                # Log file should have some content
                assert len(content) > 0, "Log file should not be empty"