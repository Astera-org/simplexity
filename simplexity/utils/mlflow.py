import json
import os
from collections.abc import Sequence

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def download_token_loss_metrics(run_id: str, metric_keys: Sequence[str]):
    """Downloads metrics of the form "validation/token_loss_{i}" for i in range(100) from a given MLflow run.

    Parameters:
        experiment_id (str): MLflow experiment ID.
        run_id (str): MLflow run ID.
        output_path (str, optional): File path to save the metrics as JSON. If None, does not save to disk.

    Returns:
        dict: Dictionary containing the token loss metrics.
    """
    client = MlflowClient()

    metrics = {}
    for key in metric_keys:
        try:
            metrics[key] = client.get_metric_history(run_id, key)
        except RestException:
            print(f"Metric {key} not found")
    return metrics


def save_metrics(metrics: dict, output_path: str | None = None) -> None:
    """Save metrics to a JSON file."""
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
