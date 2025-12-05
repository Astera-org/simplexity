# Simplexity

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Astera-org/simplexity)

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

#### Advanced Training Metrics (PyTorch)

The `TrainingMetricTracker` under `simplexity.metrics.metric_tracker` keeps
instantaneous (current loss, learning rate, parameter/gradient norms) and
cumulative metrics (tokens processed, running average loss, distance from
initialization, cumulative parameter-update norms) in sync with a PyTorch
optimizer. The tracker is wired into
`simplexity.training.train_pytorch_model.train`, but it can also be used directly
when writing custom training loops:

```python
import torch
from simplexity.metrics.metric_tracker import MetricTracker

model = build_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10_000)
tracker = MetricTracker(
    model,
    optimizer,
    optimal_loss=0.05,
    lr_schedule_fn=lambda step: lr_schedule.get_last_lr()[0],
)

for step, batch in enumerate(train_loader, start=1):
    logits = model(batch["inputs"])
    loss = criterion(logits, batch["labels"])
    if tracker.initial_loss is None:
        tracker.record_initial_loss(float(loss.detach().item()))

    loss.backward()
    optimizer.step()
    lr_schedule.step()
    optimizer.zero_grad(set_to_none=True)

    metrics = tracker.update(
        step=step,
        loss=loss,
        tokens_in_batch=int(batch["labels"].numel()),
    )
    logger.log_metrics(step, metrics)
```

Each call returns a flat dictionary with keys such as `step/tokens`, `cum/tokens`,
`loss/step`, `loss/ma`, `lr/step`, `grads/l2_norm/step`, `params/update_l2_norm/cum`,
and `params/distance_from_init`, ready to be sent to any `Logger` implementation.
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
