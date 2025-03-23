# Simplexity

A library for exploring sequence prediction models from a Computational Mechanics perspective.

## Codebase Structure

```
simplexity/
  ├── data/                      # generated data
  ├── notebooks/                 # workflow and analysis examples
  ├── simplexity/                # core library code
  │   ├── configs/               # hydra configuration files
  │   ├── generative_processes/  # generative process classes
  │   ├── logging/               # parameter and metric logging
  │   ├── persistence/           # model loading and checkpointing
  │   ├── prediction_models/     # prediction model classes
  │   ├── training/              # model training functions
  │   ├── train_model.py         # entrypoint for training models
  └── tests/                     # unit tests
```

## Usage

### Installation

[Install UV](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies:

```bash
uv sync
```

to include the dev dependencies, run:

```bash
uv sync --extra dev
```

This should create a new python environment at `.venv` and install the dependencies.

### Training

Model training is configured using [Hydra](https://hydra.cc/) with config files specified in the `simplexity/configs` directory. To train a model using the configuration specified in `simplexity/configs/train_model.yaml`, run:

```bash
uv run python simplexity/train_model.py
```
