# ProductGenerator Documentation

## Overview

The `ProductGenerator` is a generative process that creates cartesian products of sequences from multiple component generators. It combines multiple independent generative processes (HMMs, GHMMs, etc.) into a single process where the output vocabulary is the product of the component vocabularies.

## Key Concepts

### Cartesian Product of Sequences

When you have two generators:
- Generator 1 with vocabulary size `V1` (e.g., 3 tokens)
- Generator 2 with vocabulary size `V2` (e.g., 4 tokens)

The ProductGenerator creates a combined generator with vocabulary size `V1 × V2` (e.g., 3 × 4 = 12 tokens).

**Important**: The vocabulary size is **multiplicative** (V1 × V2), not additive (V1 + V2).

### Token Encoding/Decoding

Each product token uniquely identifies the component tokens:
```
Product Token 0  = [Component1: 0, Component2: 0]
Product Token 1  = [Component1: 0, Component2: 1]
Product Token 2  = [Component1: 0, Component2: 2]
...
Product Token 11 = [Component1: 2, Component2: 3]
```

## Implementation

### Basic Usage

```python
from simplexity.generative_processes.builder import build_product_generator

# Create a product generator from component configurations
configs = [
    {'type': 'hmm', 'process_name': 'mess3', 'x': 0.15, 'a': 0.6},
    {'type': 'ghmm', 'process_name': 'tom_quantum', 'alpha': 1.0, 'beta': 1.0}
]

product_gen = build_product_generator(configs)

# Generate sequences
import jax
key = jax.random.PRNGKey(0)
state = product_gen.initial_state
final_state, observations = product_gen.generate(state, key, sequence_length=10, return_all_states=False)
```

### State Representation

The `ProductState` maintains factored states for efficiency:
```python
class ProductState(NamedTuple):
    factored_states: list[jax.Array]  # List of individual generator states
```

The product state (kronecker product) is computed on-demand when needed for probability calculations.

## Configuration Files

### Product Generator Configs

Example configurations are provided in `simplexity/configs/generative_process/`:

#### `product_mess3_zero_one.yaml`
```yaml
name: product_mess3_zero_one
vocab_size: 6  # 3 * 2 = 6
instance:
  _target_: simplexity.generative_processes.builder.build_product_generator
  component_configs:
    - type: hmm
      process_name: mess3
      x: 0.15
      a: 0.6
    - type: hmm
      process_name: zero_one_random
      p: 0.7
bos_token: 6  # vocab_size is used as BOS token
eos_token:
```

#### `product_mess3_tom_quantum.yaml`
```yaml
name: product_mess3_tom_quantum
vocab_size: 12  # 3 * 4 = 12
instance:
  _target_: simplexity.generative_processes.builder.build_product_generator
  component_configs:
    - type: hmm
      process_name: mess3
      x: 0.15
      a: 0.6
    - type: ghmm
      process_name: tom_quantum
      alpha: 1.0
      beta: 1.0
bos_token: 12  # vocab_size is used as BOS token
eos_token:
```

## Examples

### Example 1: Mess3 × Zero-One Random

- **Mess3**: 3-state HMM with vocab_size = 3
- **Zero-One Random**: 2-state HMM with vocab_size = 2
- **Product**: vocab_size = 6 (3 × 2)

Token mapping:
```
Token 0: [mess3=0, zero_one=0]
Token 1: [mess3=0, zero_one=1]
Token 2: [mess3=1, zero_one=0]
Token 3: [mess3=1, zero_one=1]
Token 4: [mess3=2, zero_one=0]
Token 5: [mess3=2, zero_one=1]
```

### Example 2: Mess3 × Tom Quantum

- **Mess3**: 3-state HMM with vocab_size = 3
- **Tom Quantum**: 3-state GHMM with vocab_size = 4
- **Product**: vocab_size = 12 (3 × 4)

This demonstrates that the product generator works with:
- Different vocabulary sizes (3 ≠ 4)
- Different process types (HMM × GHMM)

## Belief States and Probability Distributions

### Factored States
Each component maintains its own belief state, stored in `ProductState.factored_states`.

### Product State
The full product state is the kronecker product of component states:
- If component 1 has `n1` states and component 2 has `n2` states
- The product has `n1 × n2` states

### Observation Probabilities
The observation probability distribution is the kronecker product of component distributions:
```python
P(product_token) = P(component1_token) × P(component2_token)
```

## BOS/EOS Tokens for Training

### How BOS Tokens Work with ProductGenerator

Beginning-of-Sequence (BOS) and End-of-Sequence (EOS) tokens are special markers used in sequence modeling:

1. **BOS Token Value**: Set to the product vocab_size
   - For mess3 × zero_one: BOS = 6 (vocab_size = 3 × 2 = 6)
   - For mess3 × tom_quantum: BOS = 12 (vocab_size = 3 × 4 = 12)

2. **Application Level**: BOS/EOS tokens are added at the **product level**, not to individual components
   - The ProductGenerator itself generates product tokens 0 through vocab_size-1
   - The training pipeline adds BOS token (vocab_size) to the beginning of sequences

3. **Training Usage**: The `generate_data_batch()` function handles BOS/EOS tokens:
```python
from simplexity.generative_processes.generator import generate_data_batch

# Generate training data with BOS token
gen_states, inputs, labels = generate_data_batch(
    gen_states,
    product_generator,
    batch_size=64,
    sequence_len=32,
    key=key,
    bos_token=product_generator.vocab_size,  # Use vocab_size as BOS
    eos_token=None  # Optional EOS token
)
```

4. **Sequence Structure with BOS**:
   - Generated sequence: `[token1, token2, token3, ...]`
   - With BOS prepended: `[BOS, token1, token2, token3, ...]`
   - Training inputs: `[BOS, token1, token2, ...]`
   - Training labels: `[token1, token2, token3, ...]`

This allows models to learn that sequences start with a special token, improving generation quality.

## Testing

### Unit Tests
Run tests with:
```bash
python -m pytest tests/generative_processes/test_product_generator.py -v
```

### Smoke Test
A comprehensive smoke test is available:
```bash
python scripts/smoke_test_product_generator.py
```

This demonstrates:
- Token encoding/decoding
- Sequence generation
- Belief state evolution
- Probability distributions

## Key Features

1. **Efficient Factored Representation**: Maintains component states separately, computing products only when needed
2. **Flexible Components**: Works with any combination of GenerativeProcess implementations
3. **Proper Probability Handling**: Correctly computes joint distributions via kronecker products
4. **Mixed Process Types**: Can combine HMMs with GHMMs or other process types
5. **Variable Vocabulary Sizes**: Components can have different vocabulary sizes

## Implementation Details

### Token Encoding
Tokens are encoded using a mixed-radix number system:
```python
product_token = Σ(component_token[i] × Π(vocab_size[j] for j > i))
```

### Token Decoding
Decoding reverses the encoding process:
```python
for each component i (in reverse):
    component_token[i] = product_token % vocab_size[i]
    product_token //= vocab_size[i]
```

### State Transitions
When a product token is observed:
1. Decode into component tokens
2. Update each component's state independently
3. The new product state is implicitly defined by the factored states

## Common Use Cases

1. **Studying Compositional Structure**: Analyze how models learn to decompose product sequences
2. **Multi-Task Learning**: Train on multiple processes simultaneously
3. **Complexity Scaling**: Create processes of controlled complexity
4. **Independence Testing**: Verify if models learn component independence

## Limitations

- Log-space computations currently convert to regular space (TODO: implement proper log-space kronecker products)
- Components must be independent (no interaction between generators)
- Memory usage scales with the product of component sizes

## Future Enhancements

- [ ] Implement proper log-space kronecker product computations
- [ ] Add support for dependent/interacting components
- [ ] Optimize memory usage for large product spaces
- [ ] Add visualization tools for product sequences