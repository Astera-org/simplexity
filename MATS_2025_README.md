# MATS 2025 Application Branch

This branch includes TransformerLens integration for the simplexity library, making it easy to train and analyze transformer models on computational mechanics processes.

## Installation

### Local Installation with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with TransformerLens support
uv sync --extra transformerlens
```

### Google Colab Installation

```python
!pip install git+https://github.com/adamimos/simplexity.git@MATS_2025_app#egg=simplexity[transformerlens]
```

## Usage

See `notebooks/train_transformerlens_mess3.ipynb` for a complete example of:
- Training a TransformerLens model on the mess3 Hidden Markov Model
- Using TransformerLens's interpretability features
- Analyzing attention patterns and model internals

## Key Features

- **Clean Integration**: TransformerLens is now an optional dependency
- **Simple Installation**: One-line pip install for Colab
- **TransformerLensWrapper**: Seamless integration between simplexity and TransformerLens
- **Full Interpretability**: Access to all TransformerLens analysis tools

## Example

```python
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.predictive_models.transformerlens_model import TransformerLensWrapper

# Create mess3 process
mess3 = build_hidden_markov_model("mess3", x=0.15, a=0.6)

# Create TransformerLens model
model = TransformerLensWrapper(
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_vocab=mess3.vocab_size
)

# Train and analyze!
```