# Simplexity

A library for exploring sequence prediction models from a Computational Mechanics perspective.

## Codebase Structure

```
simplexity/
  ├── data/                      # generated data
  ├── notebooks/                 # workflow and analysis examples
  ├── simplexity/                # core library code
  │   ├── configs/               # hydra configuration files
  │   ├── data_structures/       # data structures
  │   ├── evaluation/            # functions for model eval/validation
  │   ├── generative_processes/  # generative process classes
  │   ├── logging/               # parameter and metric logging
  │   ├── persistence/           # model loading and checkpointing
  │   ├── predictive_models/     # predictive model classes
  │   ├── training/              # model training functions
  │   ├── utils/                 # utils
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

An experiment comprised of several runs can be executed to perform a hyperparameter search or otherwise run multiple trials with different parameters using [Optuna](https://optuna.org/). To run an experiment using the configuration specified in `simplexity/configs/experiment.yaml`, run:

```bash
uv run python simplexity/run_experiment.py --multirun
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

[AWS configuration and credential files](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html) can be used for authentication and settings. Authentication credentials should be specified in `~/.aws/credentials`. Settings like `region`, `output`, `endpoint_url` should be specified in `~/.aws/config`. Multiple different profiles can be defined and the specific profile to use can be specified in the `aws` section of the `.ini` file.

### Loading From MLflow

Simplexity provides a high‑level loader to reconstruct models and read run data from MLflow.

Quick start:

```python
from simplexity.loaders import ExperimentLoader

# Use your MLflow run ID and tracking URI (e.g., Databricks)
loader = ExperimentLoader.from_mlflow(
    run_id="<RUN_ID>",
    tracking_uri="databricks",  # or None to rely on env MLFLOW_TRACKING_URI
)

# Load saved Hydra config and inspect
cfg = loader.load_config()
print("Model target:", cfg.predictive_model.instance._target_)

# Discover checkpoints and load the latest model
print("Available checkpoints:", loader.list_checkpoints())
model = loader.load_model(step="latest")

# Fetch metrics as a tidy pandas DataFrame
df = loader.load_metrics(pattern="validation/*")  # glob filter optional
print(df.head())
```

Notes:

- PyTorch models: if your run used a PyTorch model (e.g., `transformer_lens.HookedTransformer`), ensure the package is installed in your environment. The loader first tries JAX’s `PredictiveModel` path, then falls back to `torch.nn.Module` and sets `model.eval()` by default.
- Persistence: the loader reconstructs the persister from the saved config. If the run has no persistence or no checkpoints, `load_model()` raises an informative error.
- S3 credentials: if your persister uses `S3Persister.from_config`, you can override the location of the `.ini` via `ExperimentLoader.from_mlflow(..., config_path="/path/to/config.ini")`.
- Metrics filtering uses glob syntax (e.g., `"validation/*"`).

See `notebooks/experiment_loader_demo.ipynb` for a runnable example.
