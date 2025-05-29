# Simplexity Cursor Rules

This directory contains a comprehensive set of cursor rules (.mdc files) designed to help LLM coding agents work effectively with the Simplexity codebase.

## Overview

These rules are organized hierarchically to enable efficient context loading. Each rule file contains focused guidance for specific aspects of the codebase, allowing LLM agents to selectively load only the information they need for a given task.

## How to Use

1. **Start with the index**: Always begin by loading `00_index.mdc` to understand the available rules
2. **Load framework overview**: For most tasks, load `01_framework_overview.mdc` for context
3. **Load specific rules**: Based on your task, load the relevant component-specific rules
4. **Apply code style**: When making changes, reference `11_code_style.mdc`

## Rule Categories

### Core Framework Rules (01-03)
- **01_framework_overview.mdc**: High-level architecture and design principles
- **02_hydra_configuration.mdc**: Configuration system patterns and best practices
- **03_experiment_workflow.mdc**: Running experiments and hyperparameter optimization

### Component Rules (04-09)
- **04_generative_processes.mdc**: Data generation components and patterns
- **05_predictive_models.mdc**: Model architectures and integration approaches
- **06_training_system.mdc**: Training loops, optimizers, and training patterns
- **07_persistence.mdc**: Model checkpointing and storage (local/S3)
- **08_evaluation.mdc**: Model evaluation and metric computation
- **09_logging.mdc**: MLflow integration and experiment tracking

### Development Rules (10-12)
- **10_testing.mdc**: Test structure and testing patterns
- **11_code_style.mdc**: Coding conventions and standards
- **12_common_patterns.mdc**: Frequently used patterns and utilities

## Best Practices for LLM Agents

### When Adding New Features

1. Load: `00_index.mdc`, `01_framework_overview.mdc`, relevant component rules, `11_code_style.mdc`
2. Follow the existing patterns in the codebase
3. Add appropriate configuration files
4. Write tests following patterns in `10_testing.mdc`
5. Use utilities from `12_common_patterns.mdc` when applicable

### When Debugging

1. Load the specific component rule for the area you're debugging
2. Check `12_common_patterns.mdc` for debugging utilities
3. Reference `10_testing.mdc` for test-based debugging approaches

### When Running Experiments

1. Load `03_experiment_workflow.mdc` for experiment patterns
2. Reference `02_hydra_configuration.mdc` for configuration overrides
3. Use `09_logging.mdc` for tracking experiments

## Key Design Principles

1. **Protocol-based interfaces**: Use Python protocols for flexibility
2. **Pure functions**: Follow JAX patterns for functional programming
3. **Configuration-driven**: Use Hydra for all configuration
4. **Type safety**: Always include type hints
5. **Comprehensive testing**: Mirror source structure in tests

## Integration with External Libraries

The framework is designed to work seamlessly with:
- **Penzai**: Direct instantiation via configuration
- **Flax**: Through wrapper functions when needed
- **Haiku**: Via transformation to pure functions
- **Custom models**: Following the PredictiveModel protocol

## Common Workflows

### Adding a New Model
1. Choose integration approach (see `05_predictive_models.mdc`)
2. Create configuration file
3. Register with Hydra if needed
4. Add tests

### Adding a New Generative Process
1. Inherit from GenerativeProcess base class
2. Implement required methods
3. Add configuration
4. Create builder if complex

### Running Hyperparameter Search
1. Configure sweep in experiment.yaml
2. Use Optuna parameters
3. Run with --multirun flag
4. Analyze results in MLflow

## Tips for Effective Use

- **Don't load all rules at once**: This defeats the purpose of modular context
- **Load rules based on task**: Only load what you need
- **Cross-reference between rules**: Rules often reference each other
- **Use the index**: It's designed to help you find the right rules quickly
- **Follow the patterns**: The codebase is consistent, learn and follow its patterns

## Maintenance

When updating the codebase:
1. Update relevant rule files to reflect changes
2. Ensure examples in rules remain accurate
3. Add new rules if introducing major new components
4. Keep the index up to date 