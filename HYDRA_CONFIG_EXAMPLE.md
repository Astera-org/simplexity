# Hydra Configuration for Factored Generators

Factored generators are now fully supported through Hydra configuration, maintaining consistency with the existing codebase patterns.

## Configuration Classes

### FactoredHmmGeneratorConfig
For generators with all HMM components (most common case):

```python
from simplexity.configs.generative_process.config import FactoredHmmGeneratorConfig

config = FactoredHmmGeneratorConfig(
    _target_="simplexity.generative_processes.builder.build_factored_hmm_generator",
    process_name="factored_generator",  # Required by base class
    component_specs=[
        {"process_name": "zero_one_random", "p": 0.8},
        {"process_name": "zero_one_random", "p": 0.2}
    ]
)
```

### FactoredGeneratorConfig  
For generators with mixed HMM/GHMM components:

```python
from simplexity.configs.generative_process.config import FactoredGeneratorConfig

config = FactoredGeneratorConfig(
    _target_="simplexity.generative_processes.builder.build_factored_generator",
    process_name="factored_generator", 
    component_specs=[
        {"process_name": "zero_one_random", "p": 0.8},
        {"process_name": "days_of_week"}
    ],
    component_types=["hmm", "ghmm"]
)
```

## Usage with typed_instantiate

```python
from simplexity.utils.hydra import typed_instantiate
from simplexity.generative_processes.factored_generator import FactoredGenerativeProcess

# Create factored generator from config
factored_gen = typed_instantiate(config, FactoredGenerativeProcess)

# Use exactly like any other GenerativeProcess
print(f"Vocab size: {factored_gen.vocab_size}")  # e.g., 4 for 2x2 components
```

## Component Specifications

Each component spec is a dictionary with:
- `"process_name"`: One of the supported process types (`"zero_one_random"`, `"days_of_week"`, etc.)  
- Additional keyword arguments specific to that process (e.g., `"p"` for coin processes)

### Available Process Names
- `"zero_one_random"` - Coin flip HMM (requires `p` parameter)
- `"days_of_week"` - Days of week GHMM  
- `"even_ones"` - Even ones HMM (requires `p` parameter)
- `"mess3"` - Mess3 HMM (requires `x`, `a` parameters)
- `"no_consecutive_ones"` - No consecutive ones HMM (requires `p` parameter)
- `"rrxor"` - RRXor HMM (requires `pR1`, `pR2` parameters)
- `"fanizza"` - Fanizza GHMM (requires `alpha`, `lamb` parameters)
- `"post_quantum"` - PostQuantum GHMM (requires `log_alpha`, `beta` parameters)  
- `"tom_quantum"` - TomQuantum GHMM (requires `alpha`, `beta` parameters)

## Example Configurations

### Two Coin Flips (Different Biases)
```python
config = FactoredHmmGeneratorConfig(
    _target_="simplexity.generative_processes.builder.build_factored_hmm_generator",
    process_name="factored_generator",
    component_specs=[
        {"process_name": "zero_one_random", "p": 0.9},  # Heavily biased toward 0
        {"process_name": "zero_one_random", "p": 0.1}   # Heavily biased toward 1
    ]
)
# Results in vocab_size = 4 with clear factorization patterns
```

### Mixed Process Types
```python  
config = FactoredGeneratorConfig(
    _target_="simplexity.generative_processes.builder.build_factored_generator",
    process_name="factored_generator",
    component_specs=[
        {"process_name": "zero_one_random", "p": 0.7},  # Binary HMM
        {"process_name": "days_of_week"}                 # 11-token GHMM
    ],
    component_types=["hmm", "ghmm"] 
)
# Results in vocab_size = 2 * 11 = 22
```

### Three-Factor Generator
```python
config = FactoredHmmGeneratorConfig(
    _target_="simplexity.generative_processes.builder.build_factored_hmm_generator", 
    process_name="factored_generator",
    component_specs=[
        {"process_name": "zero_one_random", "p": 0.8},
        {"process_name": "even_ones", "p": 0.6}, 
        {"process_name": "no_consecutive_ones", "p": 0.4}
    ]
)
# Results in vocab_size = 2 * 2 * 2 = 8
```

## Integration with Existing Training Code

Factored generators work seamlessly with existing training pipelines:

```python
# In your main config, replace:
# training_data_generator:
#   name: zero_one_random
#   vocab_size: 2
#   instance:
#     _target_: simplexity.generative_processes.builder.build_hidden_markov_model
#     process_name: zero_one_random
#     p: 0.5

# With:
training_data_generator:
  name: factored_generator  
  vocab_size: 4  # 2 * 2
  instance:
    _target_: simplexity.generative_processes.builder.build_factored_hmm_generator
    process_name: factored_generator
    component_specs:
      - process_name: zero_one_random
        p: 0.8
      - process_name: zero_one_random  
        p: 0.2
```

## Benefits

1. **Drop-in Replacement**: Works with existing training/evaluation infrastructure
2. **Compositionality**: Mix and match different generative processes
3. **Interpretability**: Each factor's contribution remains visible
4. **Scalability**: Add complexity without exponential state growth
5. **Research Flexibility**: Easy experimentation with different factor combinations

The Hydra configuration maintains the same patterns and conventions as single generators while enabling powerful compositional modeling capabilities.