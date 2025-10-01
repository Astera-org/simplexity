# TransformerLens + mess3 Training Example

A minimal, well-documented example demonstrating how to train a TransformerLens model on synthetic data using the Simplexity framework. All artifacts including model checkpoints are uploaded to MLflow.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev --extra pytorch

# Train the model
uv run python examples/basic_mess3/train.py

# Train with custom settings
uv run python examples/basic_mess3/train.py training.num_steps=1000 training.batch_size=64
```

## What This Example Shows

1. **Training a TransformerLens model** - A small 2-layer, 2-head transformer
2. **Using synthetic data** - Generated from mess3 (a Hidden Markov Model)
3. **Device auto-detection** - Works on CPU, CUDA, and Apple Silicon (MPS)
4. **MLflow artifact storage** - Checkpoints, configs, and dependencies uploaded to MLflow
5. **Configuration with Hydra** - Clean YAML-based configuration
6. **Config resolution utilities** - Proper sequence length and vocab size computation

## Model Architecture

- **2 layers** with 2 attention heads each
- **64-dimensional** residual stream (d_model)
- **32 dimensions** per attention head (d_head)
- **Context window** of 6 tokens (n_ctx)
- **Vocabulary size** of 4 (mess3's 3 tokens + BOS token)

## Data: The mess3 Process

mess3 is a 3-state Hidden Markov Model that outputs tokens {0, 1, 2}:
- **α = 0.85**: Self-transition probability (stays in same state)
- **x = 0.05**: Cross-transition probability (switches to other states)
- **BOS token = 3**: Beginning-of-sequence marker

The process has 3 states, each deterministically outputting one of {0, 1, 2}.

## Sequence Length Math

The data generator uses next-token prediction, where:
- `inputs = tokens[:-1]` (all but last)
- `labels = tokens[1:]` (all but first)

### With BOS Token

When using a BOS token, the relationship between generator sequence length and model context length is:

```
model_n_ctx = generator_seq_len - 1 + 1  (where the +1 is from BOS)
generator_seq_len = model_n_ctx
```

In this example:
- `generator_seq_len = 6` → generates 6 tokens + prepends BOS → 7 total tokens
- After splitting: `inputs[0:6]` and `labels[0:6]` → both have 6 tokens
- Model sees 6 tokens (matching its `n_ctx = 6`)

### Config Resolution Utilities

The `simplexity.utils.config_resolution` module provides utilities to compute these relationships:

```python
from simplexity.utils.config_resolution import (
    compute_generator_sequence_length,
    compute_model_context_length,
    compute_model_vocab_size
)

# Given model n_ctx=6 and use_bos=True
generator_seq_len = compute_generator_sequence_length(model_n_ctx=6, use_bos=True)  # Returns 6

# Given generator_seq_len=6 and use_bos=True
model_n_ctx = compute_model_context_length(generator_seq_len=6, use_bos=True)  # Returns 6

# Given generator vocab_size=3, with BOS
model_vocab_size = compute_model_vocab_size(generator_vocab_size=3, use_bos=True, use_eos=False)  # Returns 4
```

## Device Selection

The training script automatically detects and uses the best available device:
- **CUDA** for NVIDIA GPUs
- **MPS** for Apple Silicon
- **CPU** as fallback

You can override with: `device=cuda` or `device=cpu`

## MLflow Artifact Storage

All artifacts are uploaded to MLflow:

### Uploaded at Start
- **Resolved config** (`config.yaml`)
- **Dependencies** (`uv.lock`, `pyproject.toml`)
- **Git information** (commit, branch, diff)

### Uploaded During Training
- **Model checkpoints** (every `checkpoint_every` steps)
  - Saved to `checkpoints/step_{N}/model.pt`
- **Metrics** (every `log_every` steps)
  - loss, grad_norm, learning_rate
  - tokens_per_second, step_time_seconds
  - memory stats (if CUDA)
- **Validation metrics** (every `validate_every` steps)
  - validation/loss, validation/perplexity

## Configuration

### Main Config

`configs/transformerlens_mess3.yaml`:
- Combines all component configs
- Sets experiment name, seed, device
- Configures logging to MLflow

### Component Configs

- `generative_process/mess3_085.yaml` - Data generation (mess3 with α=0.85)
- `predictive_model/transformer_lens_2L2H.yaml` - Model architecture
- `training/transformerlens.yaml` - Training hyperparameters
- `training/optimizer/pytorch_adam.yaml` - Optimizer settings
- `evaluation/transformerlens.yaml` - Validation settings
- `logging/mlflow_logger.yaml` - MLflow configuration

### Hydra Overrides

Override any config value from the command line:

```bash
# Change training duration
uv run python examples/basic_mess3/train.py training.num_steps=500

# Change batch size and learning rate
uv run python examples/basic_mess3/train.py training.batch_size=64 training.optimizer.instance.lr=0.0001

# Change device
uv run python examples/basic_mess3/train.py device=cpu

# Change seed
uv run python examples/basic_mess3/train.py seed=123

# Combine multiple overrides
uv run python examples/basic_mess3/train.py \
    training.num_steps=1000 \
    training.batch_size=64 \
    training.optimizer.instance.lr=0.0005 \
    device=cuda \
    seed=42
```

## File Structure

```
examples/basic_mess3/
├── train.py                           # Main training script
├── README.md                          # This file
└── configs/
    ├── transformerlens_mess3.yaml     # Main config
    ├── generative_process/
    │   └── mess3_085.yaml             # Data generation
    ├── predictive_model/
    │   └── transformer_lens_2L2H.yaml # Model architecture
    ├── training/
    │   ├── transformerlens.yaml       # Training settings
    │   └── optimizer/
    │       └── pytorch_adam.yaml      # Optimizer
    ├── evaluation/
    │   └── transformerlens.yaml       # Validation settings
    └── logging/
        └── mlflow_logger.yaml         # MLflow configuration
```

## Key Implementation Details

### Checkpoint Upload

Checkpoints are saved to a temporary directory and uploaded to MLflow inline (not using the persistence abstraction):

```python
# In train.py
if logger and training_cfg.checkpoint_every and step % training_cfg.checkpoint_every == 0:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / f"step_{step}" / "model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        logger._client.log_artifact(logger._run_id, str(checkpoint_path), artifact_path="checkpoints")
```

**Note**: In the future, this will be extracted to `simplexity.persistence.mlflow_persister.MLflowPersister`.

### Device Resolution

Uses `simplexity.utils.pytorch_utils.resolve_device()` for auto-detection:

```python
from simplexity.utils.pytorch_utils import resolve_device

device_str = config_dict.get("device", "auto")
config_dict["device"] = resolve_device(device_str)  # "cuda", "mps", or "cpu"
```

### Validation

The training script includes validation to ensure config correctness:

```python
use_bos = cfg.training_data_generator.bos_token is not None
use_eos = cfg.training_data_generator.eos_token is not None
expected_n_ctx = compute_model_context_length(cfg.training.sequence_len, use_bos)
expected_vocab_size = compute_model_vocab_size(training_data_generator.vocab_size, use_bos, use_eos)

print(f"Model n_ctx: {model_config.n_ctx}, Expected: {expected_n_ctx}")
print(f"Model d_vocab: {model_config.d_vocab}, Expected: {expected_vocab_size}")
```

## Troubleshooting

### Import Errors

If you see `ImportError: No module named 'transformer_lens'`:
```bash
uv sync --extra pytorch
```

### MLflow Connection

If MLflow fails to connect, check:
1. `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables
2. MLflow tracking URI in `configs/logging/mlflow_logger.yaml`

### Device Errors

If you see device mismatch errors:
- Try `device=cpu` to force CPU execution
- Check CUDA/MPS availability with the resolve_device utility
