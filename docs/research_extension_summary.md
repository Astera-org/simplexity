# Research Extension Summary

This document provides a quick reference for extending the Simplexity framework for research experiments.

## Key Documents

1. **[Research Extension Guide](research_extension_guide.md)** - Comprehensive guide with principles and patterns
2. **[Pattern Analysis](research_extension_patterns.md)** - Detailed analysis of successful extension patterns
3. **[Python Template](research_extension_template.py)** - Ready-to-use Python template for your experiments
4. **[Config Template](research_extension_config_template.yaml)** - YAML configuration template

## Quick Start

### 1. Set Up Your Project Structure
```bash
your-research-project/
├── your_module/
│   ├── configs/
│   ├── your_components.py
│   └── train.py
├── tests/
├── pyproject.toml
└── README.md
```

### 2. Core Extension Pattern
```python
# 1. Create custom component extending Equinox
class YourComponent(eqx.Module):
    def process(self, data, key):
        # Your logic here

# 2. Make it configurable
@dataclass
class YourComponentConfig:
    _target_: str = "your_module.YourComponent"
    
# 3. Integrate with training
def train(model, your_component, ...):
    # Use existing + custom components

# 4. Wire up with Hydra
@hydra.main(config_path="configs", config_name="experiment.yaml")
def run_experiment(cfg):
    # Instantiate and run
```

### 3. Configuration Pattern
```yaml
defaults:
  - your_component: default
  - generative_process@data_gen: existing_process
  - predictive_model: transformer
  - persistence: local
  - logging: mlflow
```

## Key Principles

1. **Extend, Don't Modify** - Build on top of Simplexity without changing core code
2. **Configuration First** - Make everything configurable via Hydra
3. **Type Safety** - Use dataclasses and type hints throughout
4. **Test Early** - Write tests for custom components
5. **Integrate Seamlessly** - Use existing abstractions and patterns

## Common Use Cases

### Adding a Custom Data Generator
- Extend `GenerativeProcess`
- Implement required methods
- Add configuration

### Custom Training Logic
- Create wrapper around existing training
- Add your custom logic
- Maintain compatibility

### New Model Architecture
- Use Penzai/Equinox patterns
- Make it configurable
- Integrate with existing training

### Custom Metrics/Logging
- Extend `Logger` interface
- Add to training loop
- Configure via Hydra

## Best Practices Checklist

- [ ] Follow Simplexity's code style
- [ ] Use JAX/Equinox patterns consistently
- [ ] Make everything configurable
- [ ] Write comprehensive tests
- [ ] Document your components
- [ ] Use type hints
- [ ] Handle random keys properly
- [ ] Log configurations and metrics
- [ ] Enable reproducibility (seeds, etc.)
- [ ] Version control properly

## Getting Help

1. Study the `simplex-research` example
2. Read Simplexity's source code
3. Follow the templates provided
4. Test incrementally
5. Ask for code review

Remember: Start small, test often, and build incrementally! 