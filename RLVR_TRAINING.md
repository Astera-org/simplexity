# RLVR Training with TRL

This document explains how to use Reinforcement Learning from Verifier Rewards (RLVR) training for the PyTorch transformer model on arithmetic tasks.

## Overview

RLVR training allows the model to learn from reward signals instead of just supervised learning. The system uses reward functions that check if the model's outputs follow the correct format and provide the correct answers for arithmetic problems.

## Installation

Install the required dependencies:

```bash
pip install --break-system-packages trl transformers accelerate chex equinox jax optax einops
```

Or add the RLVR extra to your project:

```toml
[project.optional-dependencies]
rlvr = ["torch", "transformers", "trl", "accelerate"]
```

## Usage

### Basic RLVR Training

To run RLVR training with the default configuration:

```bash
python3 simplexity/run_rlvr.py
```

### Custom Configuration

Use a specific configuration file:

```bash
python3 simplexity/run_rlvr.py --config-name=train_rlvr_test
```

### Configuration Options

The RLVR training can be configured using YAML files in `simplexity/configs/`. Key configuration files:

- `train_rlvr_model.yaml`: Main experiment configuration
- `train_rlvr_test.yaml`: Test/debug configuration
- `training/rlvr_small.yaml`: Small-scale training parameters
- `training/rlvr_large.yaml`: Large-scale training parameters

## Reward Functions

The system includes two main types of rewards:

1. **Boxed Answer Reward**: Checks if the output has the correct format (`= answer <eoe>`)
2. **Correct Answer Reward**: Checks if the answer is mathematically correct
3. **Combined Reward**: Weighted combination of both rewards

## Key Components

- `simplexity/training/reward_functions.py`: PyTorch reward function implementations
- `simplexity/training/rlvr_dataset.py`: Dataset classes for RLVR training
- `simplexity/training/train_rlvr_model.py`: Core RLVR training logic
- `simplexity/run_rlvr.py`: Main training script

## Configuration Parameters

### Training Parameters
- `num_epochs`: Number of training epochs
- `samples_per_epoch`: Number of samples per epoch
- `max_prompt_length`: Maximum length of input prompts
- `max_generation_length`: Maximum length of generated sequences
- `complexity_range`: Range of arithmetic complexity (e.g., [1, 3])

### PPO Parameters
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Training batch size
- `mini_batch_size`: PPO mini-batch size
- `ppo_epochs`: Number of PPO epochs per update
- `cliprange`: PPO clipping range
- `target_kl`: Target KL divergence

### Generation Parameters
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter

### Reward Parameters
- `reward_type`: Type of reward ("boxed", "correct", or "combined")
- `boxed_weight`: Weight for format reward
- `correct_weight`: Weight for correctness reward

## Example Configuration

```yaml
# train_rlvr_custom.yaml
defaults:
  - _self_
  - generative_process@training_data_generator: rpn_arithmetic
  - predictive_model: pytorch_transformer
  - logging: mlflow_logger
  - training@rlvr_training: rlvr_small

seed: 123
experiment_name: custom_rlvr_experiment
run_name: ${now:%Y-%m-%d_%H-%M-%S}_${experiment_name}_${seed}
```

## Monitoring

The system integrates with MLflow for logging training metrics:
- Reward statistics (mean, std)
- Policy loss
- Training progress

## Notes

- The current implementation uses a simplified policy gradient approach instead of full PPO due to TRL integration complexity
- The system is designed to work with the existing arithmetic process framework
- JAX-PyTorch conversion is handled automatically for data generation

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: The system automatically detects and uses GPU if available
3. **Memory Issues**: Reduce batch sizes in the configuration
4. **JAX Key Issues**: The system handles JAX-PyTorch conversions automatically