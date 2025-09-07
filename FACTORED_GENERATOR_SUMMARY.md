# Factored Generator Implementation Summary

## Overview
Successfully implemented a factored generative process that combines multiple independent HMMs/GHMMs using outer product token generation. The implementation provides a complete drop-in replacement for existing single generators in training pipelines.

## Key Features

### 🏗️ Architecture
- **Independent Components**: Each factor (HMM/GHMM) generates tokens independently
- **Outer Product Combination**: Token sequences are combined via tuples then mapped to composite tokens
- **Factored State Tracking**: Maintains independent belief states per component for efficiency
- **Full GenerativeProcess Compatibility**: Implements all abstract methods from base class

### 🔧 Core Functionality  
- **Token Generation**: `factor1=[0,1,1,0] + factor2=[3,2,1,2] → [(0,3),(1,2),(1,1),(0,2)] → [A,B,C,D]`
- **Vocab Size**: Product of component vocab sizes (e.g., 2×2=4, 2×11=22)
- **Bijective Mapping**: Perfect conversion between tuples and composite tokens
- **Independent Beliefs**: Each component's state evolves independently

### 🚀 Training Integration
- **Plug-and-Play**: Works seamlessly with existing `generate_data_batch`
- **Correct Shapes**: Generates proper (batch_size, sequence_len) tensors for training
- **Drop-in Replacement**: Can replace any single HMM/GHMM in training code
- **All Interfaces**: probability(), log_probability(), observation_probability_distribution() all work

### 🛠️ Builder Support
- **Easy Construction**: `build_factored_hmm_generator([("coin", {"p": 0.7}), ("coin", {"p": 0.4})])`
- **Mixed Types**: Support for HMM + GHMM combinations via `component_types` parameter
- **Configuration Ready**: Integrates with existing builder pattern

## Usage Examples

### Basic Usage
```python
from simplexity.generative_processes.builder import build_factored_hmm_generator

# Create factored generator with 2 coin flips
factored_gen = build_factored_hmm_generator([
    ("coin", {"p": 0.8}),  # Factor 1: biased toward 0
    ("coin", {"p": 0.2})   # Factor 2: biased toward 1  
])

print(f"Vocab size: {factored_gen.vocab_size}")  # Output: 4 (2×2)
```

### Training Integration  
```python
# Replace this:
# hmm = build_hidden_markov_model("coin", p=0.5)

# With this:
factored_gen = build_factored_hmm_generator([
    ("coin", {"p": 0.7}),
    ("coin", {"p": 0.3})
])

# Everything else stays the same!
gen_states, inputs, labels = generate_data_batch(
    gen_states, factored_gen, batch_size, sequence_len, key
)
```

### Mixed Component Types
```python
from simplexity.generative_processes.builder import build_factored_generator

factored_gen = build_factored_generator([
    ("coin", {"p": 0.8}),
    ("days_of_week", {})
], component_types=["hmm", "ghmm"])
```

## Implementation Details

### File Structure
- `simplexity/generative_processes/factored_generator.py` - Core implementation
- `simplexity/generative_processes/builder.py` - Builder functions added
- Full test coverage with human-readable smoke tests

### Mathematical Foundation
- **Independence**: P(composite_token | factored_state) = ∏ P(component_token_i | component_state_i)
- **Sequence Probability**: P(sequence) = ∏ P(component_sequence_i) for all components
- **State Evolution**: Each component state transitions independently based on its token

### Performance Characteristics
- **Efficient**: O(V1 × V2 × ... × VN) vocab size scaling
- **JAX Optimized**: Full JAX compilation support with @eqx.filter_jit
- **Memory Efficient**: Independent state tracking avoids exponential state space

## Testing Results

All smoke tests pass showing:
- ✅ Core class initialization and vocab size calculation  
- ✅ Outer product token generation with perfect decomposition
- ✅ Independent state transitions maintaining factorization
- ✅ Complete GenerativeProcess interface (probability calculations consistent)
- ✅ Training pipeline compatibility (proper batch shapes and vocab ranges)
- ✅ Builder support with easy instantiation patterns

## Research Impact

This implementation enables:
1. **Multi-factor modeling**: Capture independent aspects of data generation
2. **Scalable complexity**: Add factors without exponential state growth
3. **Interpretable decomposition**: Each factor's contribution remains visible  
4. **Plug-and-play research**: Easy substitution in existing training pipelines
5. **Compositional generation**: Mix different types of generative components

The factored generator maintains the simplicity of the existing codebase while adding powerful compositional modeling capabilities for research applications.