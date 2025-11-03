# Claude Code Guidelines for Simplexity

This document provides guidance for Claude when working on the Simplexity codebase, a JAX-based computational mechanics library for sequence prediction models.

## Project Overview

Simplexity is a research-oriented machine learning library focused on computational mechanics perspectives of sequence prediction. The codebase uses JAX/Equinox for neural network implementations and follows functional programming patterns.

## Code Style and Conventions

### Python Style
- **Line Length**: Maximum 120 characters
- **Python Version**: 3.12+
- **Formatting**: Use `ruff format` for automatic formatting
- **Linting**: Follow ruff rules defined in `pyproject.toml`
- **Type Annotations**: Always use type hints for function parameters and return values
- **Docstrings**: Follow Google style docstrings for all public functions and classes
- **Import Order**: Managed by ruff's isort rules (standard library, third-party, local)

### Code Quality Standards
1. **Type Safety**: All code must pass `pyright` type checking in standard mode
2. **Testing**: Write pytest tests for new functionality, maintain high coverage
3. **No Comments**: Avoid inline comments; code should be self-documenting through clear naming and structure
4. **Functional Style**: Prefer functional programming patterns, especially when working with JAX

### Naming Conventions
- **Files**: Use snake_case for all Python files
- **Classes**: Use PascalCase for classes
- **Functions/Methods**: Use snake_case
- **Constants**: Use UPPER_SNAKE_CASE
- **Type Variables**: Use PascalCase (e.g., `State`, `Model`)

## Testing Guidelines

### Test Structure
- Place tests in the `tests/` directory mirroring the source structure
- Use pytest fixtures for common test setup
- Test files must start with `test_`
- Test functions must start with `test_`

### Test Patterns
```python
@pytest.fixture
def model_fixture() -> Model:
    return build_model(...)

def test_functionality(model_fixture: Model):
    # Arrange
    input_data = ...
    
    # Act
    result = model_fixture.process(input_data)
    
    # Assert
    chex.assert_trees_all_close(result, expected)
```

### JAX-specific Testing
- Use `chex` for JAX array assertions
- Test with different random seeds
- Verify shape and dtype consistency

### Coverage Requirements
- Minimum test coverage: 80% (enforced in CI and locally)
- Coverage is automatically checked when running `pytest`
- HTML coverage reports are generated in `htmlcov/` directory
- CI will fail if coverage drops below the threshold

## Architecture Patterns

### Module Structure
```
simplexity/
├── configs/          # Hydra configuration files
├── data_structures/  # Core data structures
├── evaluation/       # Model evaluation functions
├── generative_processes/  # Generative model implementations
├── logging/          # Logging utilities
├── persistence/      # Model checkpointing
├── predictive_models/  # Neural network models
├── training/         # Training loops and utilities
└── utils/           # Helper functions
```

### Design Patterns
1. **Protocol Classes**: Use `typing.Protocol` for defining interfaces
2. **Abstract Base Classes**: Use `eqx.Module` with `@abstractmethod` for base classes
3. **Builder Pattern**: Provide builder functions for complex object construction
4. **Separation of Concerns**: Keep model definitions, training logic, and evaluation separate

## JAX/Equinox Best Practices

1. **Pure Functions**: Keep functions pure and side-effect free
2. **Vectorization**: Use `eqx.filter_vmap` for batch processing
3. **Random Keys**: Always split PRNG keys appropriately
4. **Tree Operations**: Use JAX tree operations for nested structures
5. **JIT Compilation**: Consider JIT compatibility when writing functions

## Dependency Management

- **Package Manager**: Use `uv` for dependency management
- **Dependencies**: Add to `pyproject.toml` in appropriate sections
- **Optional Dependencies**: Use extras for optional features (aws, cuda, dev, mac, pytorch)
- **Version Pinning**: Specify minimum versions with `>=`

## CI/CD Requirements

Before submitting code, ensure it passes:
1. `uv run --extra dev ruff check` - Linting
2. `uv run --extra dev ruff format --check` - Formatting
3. `uv run --extra dev --extra pytorch pyright` - Type checking  
4. `uv run --extra dev --extra pytorch pytest` - Tests with coverage (must meet 80% threshold)

## Common Commands

```bash
# Install dependencies
uv sync --extra dev

# Run linting
uv run --extra dev ruff check

# Format code
uv run --extra dev ruff format

# Type check
uv run --extra dev --extra pytorch pyright

# Run tests
uv run --extra dev --extra pytorch pytest

# Run tests with coverage (will fail if below 80%)
uv run --extra dev --extra pytorch pytest

# View HTML coverage report (generated in htmlcov/)
uv run --extra dev --extra pytorch pytest
# Then open htmlcov/index.html in a browser

# Train a model
uv run python simplexity/train_model.py
```

## Pull Request Guidelines

When reviewing or creating PRs:
1. Ensure all CI checks pass
2. Maintain or improve test coverage
3. Follow existing patterns and conventions
4. Keep changes focused and atomic
5. Update relevant documentation

## Security Considerations

- Never commit credentials or API keys
- Use environment variables or config files for sensitive data
- Follow AWS best practices when using S3 persistence
- Validate all external inputs

## Performance Considerations

1. Profile JAX code for bottlenecks
2. Prefer vectorized operations over loops
3. Use appropriate data structures from `simplexity.data_structures`
4. Consider memory usage with large models
5. Leverage GPU acceleration when available

## Documentation Standards

- Keep documentation concise and technical
- Focus on "why" rather than "what"
- Update README.md for user-facing changes
- Use type hints as primary documentation
- Provide usage examples in docstrings when helpful