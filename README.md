# Epsilon Transformers

This codebase contains the code for analyzing transformers from a Computational Mechanics point of view.

## Codebase Structure

epsilon_transformers/
  ├── epsilon_transformers/
  │   ├── analysis/
  │   ├── process/
  │   ├── training/
  │   └── visualization/
  ├── scripts/
  │   ├── train.py        # CLI entry point for training
  │   └── analyze.py      # CLI entry point for analysis
  └── setup.py

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

## Dev

For formatting, type checking, and testing you can run the corresponding script in `scripts/`
