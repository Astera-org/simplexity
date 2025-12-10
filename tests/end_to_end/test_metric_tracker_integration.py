"""Test metric tracker integration without full demo dependencies."""

import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

import simplexity

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")
CONFIG_DIR = str(Path(__file__).parent / "configs")


class SimpleModel(nn.Module):
    """Simple model for testing metric tracker."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.embedding(x)
        return self.linear(x)


@simplexity.managed_run(strict=False, verbose=False)
def _run_metric_tracker_test(_cfg, components: simplexity.Components) -> None:  # pylint: disable=unused-argument
    """Run the metric tracker integration test."""
    SIMPLEXITY_LOGGER.info("Testing metric tracker integration")

    # Check that metric tracker was instantiated
    assert components.metric_trackers is not None, "Metric trackers should be instantiated"
    metric_tracker = components.get_metric_tracker()
    assert metric_tracker is not None, "Metric tracker should be available"

    SIMPLEXITY_LOGGER.info("Metric tracker type: %s", type(metric_tracker))

    # Create simple model and optimizer for testing
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Override the model and optimizer in the metric tracker
    metric_tracker.model = model
    metric_tracker.optimizer = optimizer

    # Run a simple training loop
    SIMPLEXITY_LOGGER.info("Running 5 test training steps")
    for step in range(5):
        # Generate random data
        inputs = torch.randint(0, 100, (4, 10))
        targets = torch.randint(0, 100, (4, 10))

        # Forward pass
        outputs = model(inputs)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metric tracker
        metric_tracker.step(tokens=inputs.numel(), loss=loss.item())

        # Get metrics
        metrics = metric_tracker.get_metrics(group="all")
        loss_val = metrics.get("loss", 0.0)
        tokens_val = metrics.get("tokens/raw", "N/A")
        SIMPLEXITY_LOGGER.info("Step %d metrics: loss=%.4f, tokens=%s", step, loss_val, tokens_val)

    SIMPLEXITY_LOGGER.info("Metric tracker integration test PASSED")


def test_metric_tracker(tmp_path: Path) -> None:
    """Test the metric tracker integration."""
    mlflow_db = tmp_path / "mlflow.db"
    mlflow_uri = f"sqlite:///{mlflow_db.absolute()}"
    overrides = [
        f"mlflow.tracking_uri={mlflow_uri}",
        f"mlflow.registry_uri={mlflow_uri}",
    ]
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        cfg = compose(config_name="test_metric_tracker.yaml", overrides=overrides)
    _run_metric_tracker_test(cfg)  # pylint: disable=no-value-for-parameter
