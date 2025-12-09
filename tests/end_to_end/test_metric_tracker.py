"""Test metric tracker integration without full demo dependencies."""

import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

# Add simplexity to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import simplexity

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)


@hydra.main(config_path="configs", config_name="test_metric_tracker.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=False)
def test_metric_tracker(cfg: DictConfig, components: simplexity.Components) -> None:
    """Test the metric tracker integration."""
    SIMPLEXITY_LOGGER.info("Testing metric tracker integration")
    
    # Check that metric tracker was instantiated
    assert components.metric_trackers is not None, "Metric trackers should be instantiated"
    metric_tracker = components.get_metric_tracker()
    assert metric_tracker is not None, "Metric tracker should be available"
    
    SIMPLEXITY_LOGGER.info(f"Metric tracker type: {type(metric_tracker)}")
    
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
        metric_tracker.update_metrics(group="all")
        
        # Get metrics
        metrics = metric_tracker.metrics()
        SIMPLEXITY_LOGGER.info(f"Step {step} metrics: loss={metrics.get('loss', 'N/A'):.4f}, tokens={metrics.get('tokens/raw', 'N/A')}")
    
    SIMPLEXITY_LOGGER.info("✅ Metric tracker integration test PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_metric_tracker()
