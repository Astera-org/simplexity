# Simplexity

A library for exploring sequence prediction models from a Computational Mechanics perspective. Simplexity is a Python library designed for researchers and developers to build, train, and analyze sequence prediction models, with a particular focus on applying principles from Computational Mechanics to understand their behavior and capabilities. It supports various generative processes and predictive model architectures.

## Codebase Structure

```
simplexity/
├── data/                      # Stores generated datasets or large data files used by the project.
├── documentation/             # Detailed documentation, guides, and LLM interaction instructions.
├── notebooks/                 # Jupyter notebooks for EDA, workflow examples, and result visualization.
├── outputs/                   # Default directory for experiment results, trained models, and logs.
├── simplexity/                # Core library code.
│   ├── configs/               # Hydra configuration files
│   ├── data_structures/       # Data structures
│   ├── evaluation/            # Functions for model eval/validation
│   ├── generative_processes/  # Generative process classes
│   ├── logging/               # Parameter and metric logging
│   ├── persistence/           # Model loading and checkpointing
│   ├── predictive_models/     # Predictive model classes
│   ├── training/              # Model training functions
│   ├── utils/                 # Utility functions
│   ├── train_model.py         # Entrypoint for training models
│   └── penzai_utils.py        # Utilities specific to Penzai integration
├── tests/                     # Unit and integration tests for the simplexity library.
└── .venv/                     # Python virtual environment managed by uv.
```

## Usage

### Installation

[Install UV](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This project requires Python 3.9+. Install the dependencies:

```bash
uv sync
```

To include the dev dependencies (for running tests, etc.), run:

```bash
uv sync --extra dev
```

This should create a new python environment at `.venv` and install the dependencies.

### Key Scripts and Workflows

*   `simplexity/train_model.py`: Main script for training a single predictive model. Uses [Hydra](https://hydra.cc/) for configuration (see `simplexity/configs/train_model.yaml` for an example).
    ```bash
    uv run python simplexity/train_model.py
    ```

*   `simplexity/run_experiment.py`: Script for running multiple training trials, often for hyperparameter optimization using [Optuna](https://optuna.org/). Uses Hydra for configuration (see `simplexity/configs/experiment.yaml` for an example).
    ```bash
    uv run python simplexity/run_experiment.py --multirun
    ```

### Running Tests

To ensure the integrity of the codebase, first install dev dependencies (`uv sync --extra dev`), then run the test suite using:

```bash
uv run pytest
```

### Model Checkpointing

The `ModelPersister` class is responsible for saving and loading model checkpoints. The `LocalPersister` class saves checkpoints to the local file system, while the `S3Persister` class saves checkpoints to an S3 bucket.
#### Using S3 Storage

The `S3Persister`, can be configured using an `.ini` file, which should have the following structure:

```ini
[aws]
profile_name = default

[s3]
bucket = your_s3_bucket_name
prefix = your_s3_prefix
```

[AWS configuration and credential files](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html) can be used for authentication and settings. Authentication credentials should be specified in `~/.aws/credentials`. Settings like `region`, `output`, `endpoint_url` should be specified in `~/.aws/config`. Multiple different profiles can be defined and the specific profile to use can be specified in the `aws` section of the `.ini` file. The S3 configuration `.ini` file itself should be placed at a known location (e.g., `~/.config/simplexity/s3_config.ini`) and its path made available to the application, potentially via an environment variable or a configuration setting within Hydra.

## Documentation

For more detailed information, including conceptual explanations, API references, and advanced usage, please refer to the [documentation/](./documentation/) directory.

Specifically for LLM agent interaction and guidance, see [documentation/6.0-For-LLM-Agents/](./documentation/6.0-For-LLM-Agents/).
