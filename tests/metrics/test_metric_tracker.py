import torch
import torch.nn as nn

import pytest

from simplexity.metrics.metric_tracker import TrainingMetricTracker


class TinyModel(nn.Module):
    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_metric_tracker_reports_expected_metrics() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tracker = TrainingMetricTracker(model=model, optimizer=optimizer, optimal_loss=0.05)

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)
    loss_fn = nn.MSELoss()
    tokens_in_batch = int(inputs.numel())
    observed_losses: list[float] = []

    for step in range(1, 3):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_tensor = loss_fn(outputs, targets)
        observed_loss = float(loss_tensor.detach().item())
        observed_losses.append(observed_loss)
        if tracker.initial_loss is None:
            tracker.record_initial_loss(observed_loss)
        loss_tensor.backward()
        optimizer.step()

        metrics = tracker.update(step=step, loss=loss_tensor, tokens_in_batch=tokens_in_batch)

        assert metrics["loss"] == pytest.approx(observed_loss)
        assert metrics["loss/avg"] == pytest.approx(sum(observed_losses) / len(observed_losses))
        assert metrics["tokens/total"] == pytest.approx(step * tokens_in_batch)
        assert metrics["lr"] == pytest.approx(0.1)
        assert metrics["lr/current_peak"] >= metrics["lr"]
        assert metrics["lr/weighted_peak"] >= metrics["lr/weighted_by_tokens"]
        assert metrics["params/l2_norm"] > 0.0
        assert metrics["params/distance_from_init"] >= 0.0
        assert metrics["params/update_l2_norm/cumulative"] >= metrics["params/update_l2_norm"]
        assert metrics["grads/l2_norm"] > 0.0
        assert metrics["loss/initial"] == tracker.initial_loss
        assert metrics["loss/optimal"] == 0.05


def test_record_initial_loss_is_idempotent() -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    tracker = TrainingMetricTracker(model=model, optimizer=optimizer)

    tracker.record_initial_loss(1.0)
    tracker.record_initial_loss(2.0)

    assert tracker.initial_loss == 1.0
