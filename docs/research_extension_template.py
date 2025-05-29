"""
Template for extending Simplexity framework for research experiments.

This template demonstrates the key patterns for creating custom components
and integrating them with the existing Simplexity framework.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
from penzai.nn.layer import Layer as PenzaiModel
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.hydra import typed_instantiate

# ============================================================================
# STEP 1: Define your custom component
# ============================================================================

class YourCustomComponent(eqx.Module):
    """
    Your custom component that extends Simplexity functionality.
    
    This example shows how to create a JAX-compatible component using Equinox.
    Replace this with your actual component logic.
    """
    
    # Define your component's attributes as class variables
    param1: jax.Array
    param2: float
    
    def __init__(self, param1_value: list[float], param2: float = 0.5):
        """Initialize your component with configuration parameters."""
        self.param1 = jnp.array(param1_value)
        self.param2 = param2
    
    def process(self, input_data: jax.Array, key: chex.PRNGKey) -> jax.Array:
        """
        Main processing method for your component.
        
        Args:
            input_data: Input array to process
            key: Random key for any stochastic operations
            
        Returns:
            Processed output array
        """
        # Your custom logic here
        return input_data * self.param1 + self.param2


# ============================================================================
# STEP 2: Create configuration dataclasses
# ============================================================================

@dataclass
class YourComponentInstanceConfig:
    """Configuration for instantiating your custom component."""
    _target_: str = "your_module.YourCustomComponent"
    param1_value: list[float] = field(default_factory=lambda: [1.0, 1.0])
    param2: float = 0.5


@dataclass
class YourComponentConfig:
    """Top-level configuration for your component."""
    name: str
    instance: YourComponentInstanceConfig


@dataclass
class ExperimentConfig:
    """
    Main configuration for your experiment.
    
    Combines your custom components with existing Simplexity components.
    """
    # Your custom component
    your_component: YourComponentConfig
    
    # Standard Simplexity components
    training_data_generator: Any  # DataGeneratorConfig
    validation_data_generator: Any  # DataGeneratorConfig
    predictive_model: Any  # ModelConfig
    persistence: Any  # PersistenceConfig
    logging: Any  # LoggingConfig
    training: Any  # TrainingConfig
    validation: Any  # ValidationConfig
    
    # Experiment metadata
    seed: int = 0
    experiment_name: str = "your_experiment"
    run_name: str = "${now:%Y-%m-%d_%H-%M-%S}_${experiment_name}_${seed}"


# ============================================================================
# STEP 3: Create your training function
# ============================================================================

def your_training_loop(
    model: PenzaiModel,
    your_component: YourCustomComponent,
    training_cfg: Any,  # TrainingConfig
    data_generator: GenerativeProcess,
    logger: Optional[Logger] = None,
    persister: Optional[ModelPersister] = None,
) -> tuple[PenzaiModel, float]:
    """
    Custom training loop that integrates your component.
    
    Args:
        model: The model to train
        your_component: Your custom component instance
        training_cfg: Training configuration
        data_generator: Data generation process
        logger: Optional logger for metrics
        persister: Optional model persister for checkpoints
        
    Returns:
        Trained model and final loss value
    """
    key = jax.random.PRNGKey(training_cfg.seed)
    
    # Your training logic here
    # This is a simplified example
    for step in range(training_cfg.num_steps):
        key, data_key, component_key = jax.random.split(key, 3)
        
        # Generate data
        # ... your data generation logic ...
        
        # Use your custom component
        # processed_data = your_component.process(data, component_key)
        
        # Training step
        # ... your training logic ...
        
        # Logging
        if logger and step % training_cfg.log_every == 0:
            metrics = {"loss": 0.0}  # Replace with actual metrics
            logger.log_metrics(step, metrics)
        
        # Checkpointing
        if persister and step % training_cfg.checkpoint_every == 0:
            persister.save_weights(model, step)
    
    return model, 0.0  # Return model and final metric


# ============================================================================
# STEP 4: Create the main experiment entry point
# ============================================================================

@hydra.main(config_path="configs", config_name="your_experiment.yaml", version_base="1.2")
def run_experiment(cfg: ExperimentConfig) -> float:
    """
    Main entry point for your experiment.
    
    This function is decorated with Hydra for configuration management.
    """
    # Convert OmegaConf to proper config if needed
    assert isinstance(cfg, DictConfig)
    
    # Initialize logger
    logger = typed_instantiate(cfg.logging.instance, Logger)
    logger.log_config(cfg)
    logger.log_params(cfg)
    
    # Initialize your custom component
    your_component = typed_instantiate(cfg.your_component.instance, YourCustomComponent)
    
    # Initialize standard components
    data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)
    
    # Cast model to PenzaiModel for type safety
    model = cast(PenzaiModel, model)
    
    # Run training with model persistence
    with typed_instantiate(cfg.persistence.instance, ModelPersister) as persister:
        model, final_metric = your_training_loop(
            model=model,
            your_component=your_component,
            training_cfg=cfg.training,
            data_generator=data_generator,
            logger=logger,
            persister=persister,
        )
    
    # Clean up
    logger.close()
    
    # Return metric for hyperparameter optimization
    return final_metric


# ============================================================================
# STEP 5: Add tests for your component
# ============================================================================

def test_your_component():
    """Test your custom component in isolation."""
    component = YourCustomComponent(param1_value=[1.0, 2.0], param2=0.5)
    
    key = jax.random.PRNGKey(0)
    input_data = jnp.ones((2,))
    output = component.process(input_data, key)
    
    # Add your assertions
    assert output.shape == input_data.shape
    # ... more tests ...


def test_integration():
    """Test your component's integration with Simplexity."""
    # Create a minimal config
    # Instantiate components
    # Run a few training steps
    # Verify expected behavior
    pass


if __name__ == "__main__":
    run_experiment() 