import argparse
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


class Metric(NamedTuple):
    """A metric with steps and values."""

    steps: jax.Array
    values: jax.Array


def download_metrics(client: MlflowClient, run_id: str, metric_keys: Sequence[str]) -> dict[str, Metric]:
    """Downloads metrics of the form "validation/token_loss_{i}" for i in range(100) from a given MLflow run."""
    metrics = {}
    for key in metric_keys:
        try:
            metric_history = client.get_metric_history(run_id, key)
            metrics[key] = Metric(
                steps=jnp.array([m.step for m in metric_history]),
                values=jnp.array([m.value for m in metric_history]),
            )
        except RestException:
            print(f"Metric {key} not found")
    return metrics


def token_position(metric_key: str) -> str:
    """Extracts the token from a metric key."""
    return metric_key.split("_")[-1]


def main(run_id: str, output_path: str):
    """Downloads metrics from a given MLflow run and saves them to a JSON file."""
    client = MlflowClient(tracking_uri="databricks")
    batch_size = int(client.get_run(run_id).data.params["training.batch_size"])
    sequence_length = int(client.get_run(run_id).data.params["training.sequence_len"])
    metric_keys = [f"validation/token_loss_{i}" for i in range(100)]
    metrics = download_metrics(client, run_id, metric_keys)
    new_metrics: dict[str, jnp.ndarray] = {token_position(k): v.values for k, v in metrics.items() if v.steps.size > 0}
    new_metrics["elapsed_training_tokens"] = next(iter(metrics.values())).steps * batch_size * sequence_length
    jnp.savez(output_path, **new_metrics, allow_pickle=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run_id", type=str, required=True)
    args.add_argument("--output_path", type=str, required=False, default="metrics.npz")
    args = args.parse_args()
    main(run_id=args.run_id, output_path=args.output_path)
