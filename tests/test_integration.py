"""Integration test for simplexity end-to-end workflow."""

import tempfile
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from penzai.nn.layer import Layer as PenzaiModel

from simplexity.configs.config import Config, validate_config
from simplexity.evaluation.evaluate_model import evaluate
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train
from simplexity.utils.hydra import typed_instantiate
from simplexity.utils.penzai import use_penzai_model


@pytest.mark.slow
def test_integration_workflow():
    """Test the complete workflow of training a model with all components."""

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        models_dir = temp_path / "models"
        logs_dir = temp_path / "logs"
        models_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        # Initialize Hydra with our test config
        config_dir = Path(__file__).parent / "configs"
        config_dir = config_dir.resolve()

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
            cfg: DictConfig = compose(config_name="integration_test.yaml")

            # Override paths to use temp directory
            cfg.persistence.instance.directory = str(models_dir)
            cfg.logging.instance.file_path = str(logs_dir / f"{cfg.run_name}.log")

            # Validate the configuration
            validate_config(cfg)

            # Step 1: Initialize logger
            logger = typed_instantiate(cfg.logging.instance, Logger)
            logger.log_config(cfg)
            logger.log_params(cfg)

            # Step 2: Build data generators
            training_data_generator = typed_instantiate(
                cfg.training_data_generator.instance, GenerativeProcess
            )
            validation_data_generator = typed_instantiate(
                cfg.validation_data_generator.instance, GenerativeProcess
            )

            # Step 3: Build model
            original_model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)

            # Step 4: Initialize persister
            persister = typed_instantiate(cfg.persistence.instance, ModelPersister)

            # Step 5: Get initial metrics before training
            evaluate_model = use_penzai_model(evaluate)
            initial_metrics = evaluate_model(
                model=original_model,
                cfg=cfg.validation,
                data_generator=validation_data_generator,
            )

            # Verify initial metrics are reasonable
            assert initial_metrics["loss"] > 0.0
            assert 0.0 <= initial_metrics["accuracy"] <= 1.0
            initial_loss = initial_metrics["loss"]

            # Step 6: Train the model
            trained_model, final_loss = train(
                original_model,
                cfg.training,
                training_data_generator,
                logger,
                cfg.validation,
                validation_data_generator,
                persister,
            )

            # Step 7: Verify training results
            assert final_loss > 0.0

            # Step 8: Get final metrics after training
            final_metrics = evaluate_model(
                model=trained_model,
                cfg=cfg.validation,
                data_generator=validation_data_generator,
            )

            # Step 9: Verify that final loss decreased
            assert final_metrics["loss"] < initial_loss, (
                f"Final loss {final_metrics['loss']:.4f} should be less than "
                f"initial loss {initial_loss:.4f}"
            )

            # Step 10: Verify that the trained model is different from the original
            # Compare model parameters to ensure they changed
            original_params = jax.tree.map(jnp.array, original_model)
            trained_params = jax.tree.map(jnp.array, trained_model)

            # Check that at least some parameters have changed
            params_changed = False

            def check_params(orig, trained):
                nonlocal params_changed
                if isinstance(orig, jnp.ndarray) and isinstance(trained, jnp.ndarray):
                    if orig.shape == trained.shape and not jnp.allclose(orig, trained, rtol=1e-6):
                        params_changed = True

            jax.tree.map(check_params, original_params, trained_params)
            assert params_changed, "Model parameters should have changed after training"

            # Step 11: Verify that checkpoints were saved
            checkpoint_files = list(models_dir.glob("*.pkl"))
            assert len(checkpoint_files) > 0, "No checkpoint files were saved"

            # Step 12: Load a checkpoint and verify it works
            loaded_model = persister.load_weights(
                original_model, cfg.training.checkpoint_every
            )
            loaded_metrics = evaluate_model(
                model=loaded_model,
                cfg=cfg.validation,
                data_generator=validation_data_generator,
            )

            # Verify loaded model performs better than untrained
            assert loaded_metrics["loss"] < initial_loss

            # Step 13: Verify log file was created and contains expected content
            log_file = logs_dir / f"{cfg.run_name}.log"
            assert log_file.exists(), "Log file was not created"

            log_content = log_file.read_text()
            assert "config" in log_content.lower()
            assert "loss" in log_content
            assert "accuracy" in log_content

            # Close logger
            logger.close()

            print(f"Integration test passed!")
            print(f"Initial loss: {initial_loss:.4f}")
            print(f"Final loss: {final_metrics['loss']:.4f}")
            print(f"Loss reduction: {(initial_loss - final_metrics['loss']) / initial_loss * 100:.1f}%")
            print(f"Final accuracy: {final_metrics['accuracy']:.4f}")