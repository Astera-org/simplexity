# BOS/EOS Token Handling Improvements

This branch (`adam/improve-bos-eos-handling`) improves how BOS (Beginning of Sequence) and EOS (End of Sequence) tokens are handled throughout the codebase.

## Key Changes

### 1. Configuration Simplification
- **Before**: Explicit token values and redundant vocab_size in generator configs
- **After**: Boolean flags (`use_bos_token`, `use_eos_token`) in generator configs
- Generator YAMLs no longer need to specify vocab_size (computed from generator)
- Token values computed automatically with no gaps in numbering

### 2. Token Value Assignment
The system now computes token values to ensure no gaps for embedding layers:
- No special tokens: vocab_size = base_vocab_size
- BOS only: BOS = base_vocab_size, model_vocab_size = base_vocab_size + 1
- EOS only: EOS = base_vocab_size, model_vocab_size = base_vocab_size + 1  
- Both tokens: BOS = base_vocab_size, EOS = base_vocab_size + 1, model_vocab_size = base_vocab_size + 2

### 3. Files Modified

#### Config Updates
- `simplexity/configs/generative_process/config.py`: Added boolean flags
- All generator YAML files: Updated to use boolean flags, removed vocab_size

#### Core Logic
- `simplexity/run.py`: Added token computation logic, passes tokens to train()
- `simplexity/training/train_model.py`: Updated to accept token parameters
- `simplexity/training/train_pytorch_model.py`: Updated to accept token parameters
- `simplexity/evaluation/evaluate_model.py`: Updated to accept token parameters
- `simplexity/evaluation/evaluate_pytorch_model.py`: Updated to accept token parameters

#### Tests
- `tests/training/test_train_model.py`: Updated to pass token parameters
- `tests/training/test_train_pytorch_model.py`: Updated to pass token parameters  
- `tests/evaluation/test_evaluate_model.py`: Updated to pass token parameters
- `tests/evaluation/test_evaluate_pytorch_model.py`: Updated to pass token parameters

## Benefits
1. **Cleaner configuration**: No need to manually calculate or specify token values
2. **No gaps in token numbering**: Ensures compatibility with embedding layers
3. **Centralized logic**: Token computation happens in one place (run.py)
4. **Backward compatible**: Existing code continues to work with the new system

## Usage Example

Before:
```yaml
name: my_process
instance:
  _target_: simplexity.generative_processes.builder.build_hidden_markov_model
  process_name: zero_one_random
  p: 0.5
vocab_size: 2
bos_token: 2
```

After:
```yaml
name: my_process
instance:
  _target_: simplexity.generative_processes.builder.build_hidden_markov_model
  process_name: zero_one_random
  p: 0.5
use_bos_token: true
use_eos_token: false
```

The system automatically computes that BOS token = 2 and model vocab_size = 3.