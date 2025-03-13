# Simplexity

A library for exploring sequence prediction models from a Computational Mechanics perspective.

## Codebase Structure

```
simplexity/
  ├── data/                      # generated data
  ├── notebooks/                 # workflow and analysis examples
  ├── simplexity/                # core library code
  │   ├── generative_processes/  # generative process classes
  │   ├── prediction_models/     # prediction model classes
  │   ├── training/              # model training functions
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
