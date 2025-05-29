# Research Extension Guide for Simplexity

This guide outlines best practices and patterns for extending the Simplexity framework for research experiments, based on successful implementation patterns.

## Core Principles

### 1. **Parallel Structure Pattern**
Create your research repository with a structure that mirrors the main simplexity repository:

```
your-research-project/
├── your_module_name/
│   ├── configs/
│   │   ├── evaluation/
│   │   ├── generative_process/
│   │   ├── logging/
│   │   ├── persistence/
│   │   ├── predictive_model/
│   │   └── training/
│   ├── your_custom_components.py
│   └── train.py
├── tests/
├── pyproject.toml
└── README.md
```

### 2. **Configuration-First Development**
Use Hydra for configuration management with these patterns:

#### a. Hierarchical Configuration
- Create YAML configs for each component type
- Use typed configuration dataclasses for validation
- Leverage Hydra's composition with defaults

Example configuration dataclass:
```python
from dataclasses import dataclass
from simplexity.configs.evaluation.config import Config as ValidationConfig
# Import other configs...

@dataclass
class Config:
    """Configuration for the experiment."""
    # Your custom components
    state_sampler: StateSamplerConfig
    
    # Reused simplexity components
    training_data_generator: DataGeneratorConfig
    predictive_model: ModelConfig
    persistence: PersistenceConfig
    logging: LoggingConfig
    training: TrainingConfig
    validation: ValidationConfig
    
    # Experiment metadata
    seed: int
    experiment_name: str
    run_name: str
```

#### b. Main Configuration File Pattern
```yaml
defaults:
 - _self_
 - your_custom_component: default_config
 - generative_process@training_data_generator: your_process
 - predictive_model: small_transformer
 - persistence: local_penzai_persister
 - logging: mlflow_logger
 - training: medium
 - evaluation@validation: small

seed: 0
experiment_name: ${predictive_model.name}_${training_data_generator.name}
run_name: ${now:%Y-%m-%d_%H-%M-%S}_${experiment_name}_${seed}
```

### 3. **Component Extension Pattern**

When creating custom components:

1. **Inherit from appropriate base classes**:
```python
from abc import abstractmethod
import equinox as eqx
import jax

class StateSampler(eqx.Module):
    """Base class for state sampling."""
    
    @abstractmethod
    def sample(self, key: chex.PRNGKey) -> jax.Array:
        """Sample a state from the process."""
        ...

class YourCustomSampler(StateSampler):
    """Your implementation."""
    # Custom attributes
    your_param: jax.Array
    
    def __init__(self, ...):
        # Initialize your component
        
    def sample(self, key: chex.PRNGKey) -> jax.Array:
        # Implement the abstract method
```

2. **Make components configurable via Hydra**:
```yaml
name: your_component
instance:
  _target_: your_module.YourCustomSampler
  param1: value1
  param2: value2
```

### 4. **Training Script Pattern**

Structure your training scripts with:

1. **Main entry point with Hydra decorator**:
```python
@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.2")
def run_experiment(cfg: Config) -> float:
    """Run the experiment."""
    # Instantiate components
    logger = typed_instantiate(cfg.logging.instance, Logger)
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)
    
    # Run training
    model, loss = train(...)
    
    return loss
```

2. **Separate training logic**:
```python
def train(
    model: PenzaiModel,
    training_cfg: TrainConfig,
    # Your custom components
    initial_state_sampler: StateSampler,
    # Standard components
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    ...
) -> tuple[PenzaiModel, float]:
    """Training loop implementation."""
```

### 5. **Integration Patterns**

#### a. Use Dependency Injection
- Use `typed_instantiate` from simplexity utilities
- Let Hydra handle object creation
- Type hint for better IDE support

#### b. Leverage Existing Components
- Reuse simplexity's loggers, persisters, and data generators
- Only create custom components for novel functionality
- Integrate seamlessly with existing abstractions

### 6. **Testing Pattern**

Write tests for your custom components:

```python
def test_your_component():
    # Test individual functionality
    component = YourComponent(...)
    result = component.method(...)
    assert expected_condition

def test_integration():
    # Test integration with simplexity
    # Create full pipeline
    # Verify expected behavior
```

### 7. **Hyperparameter Tuning Pattern**

For hyperparameter optimization:

1. Create a separate config:
```yaml
defaults:
  - train  # Inherit from main config
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    params:
      train.optimizer.instance.learning_rate: tag(log, interval(1e-4, 1e-1))
      train.batch_size: choice(16, 32, 64, 128)
```

2. Create a tuning script that returns the optimization metric:
```python
@hydra.main(config_path="configs", config_name="hyperparam_tune.yaml")
def run_experiment(cfg: Config) -> float:
    # ... training code ...
    return loss  # Return metric to optimize
```

## Best Practices

1. **Version Control**
   - Use git+ssh dependencies for internal packages
   - Pin versions for reproducibility

2. **Documentation**
   - Document your custom components with docstrings
   - Explain non-obvious design decisions
   - Keep README updated with usage instructions

3. **Code Quality**
   - Use type hints throughout
   - Follow the same linting rules as simplexity
   - Use equinox/JAX patterns consistently

4. **Modularity**
   - Keep custom components focused and single-purpose
   - Make components reusable across experiments
   - Avoid tight coupling with specific experiments

5. **Configuration Management**
   - Use meaningful names for configs
   - Leverage variable interpolation (${...})
   - Keep defaults sensible for quick experimentation

## Example Workflow

1. **Identify what needs to be extended**
   - New data generation process?
   - Custom model architecture?
   - Novel training procedure?

2. **Create minimal custom component**
   - Inherit from appropriate base class
   - Implement required methods
   - Add configuration

3. **Integrate with existing pipeline**
   - Add to config hierarchy
   - Update main config with your component
   - Use typed_instantiate for creation

4. **Test thoroughly**
   - Unit test the component
   - Integration test with simplexity
   - Validate expected behavior

5. **Document and share**
   - Update README
   - Add usage examples
   - Consider contributing back to simplexity

## Common Pitfalls to Avoid

1. **Don't reinvent the wheel** - Check if simplexity already has what you need
2. **Don't break abstractions** - Work within the established patterns
3. **Don't hardcode values** - Make everything configurable
4. **Don't skip tests** - Test custom components thoroughly
5. **Don't forget reproducibility** - Set seeds, log configs, save checkpoints

This guide should help you extend simplexity effectively for your research needs while maintaining code quality and reproducibility. 