"""Test metric tracker integration without full demo dependencies."""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

# Add simplexity to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import simplexity  # pylint: disable=wrong-import-position

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


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


@hydra.main(config_path="configs", config_name="test_metric_tracker.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=False)
def test_metric_tracker(_cfg: DictConfig, components: simplexity.Components) -> None:  # noqa: PT019
    """Test the metric tracker integration."""
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

    SIMPLEXITY_LOGGER.info("✅ Metric tracker integration test PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_metric_tracker()  # pylint: disable=no-value-for-parameter
