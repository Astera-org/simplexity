"""Tests for custom learning rate schedulers."""
# pylint: disable=protected-access

import pytest
import torch
from torch.optim import SGD

from simplexity.lr_schedulers import WindowedReduceLROnPlateau


@pytest.fixture
def optimizer() -> SGD:
    """Create a simple optimizer for testing."""
    model = torch.nn.Linear(10, 1)
    return SGD(model.parameters(), lr=0.1)


class TestWindowedReduceLROnPlateau:
    """Tests for WindowedReduceLROnPlateau scheduler."""

    def test_window_accumulation(self, optimizer: SGD):
        """Test that losses accumulate in the window."""
        scheduler = WindowedReduceLROnPlateau(optimizer, window_size=5, update_every=1)

        for _i in range(3):
            scheduler.step(1.0)

        assert len(scheduler._loss_window) == 3
        assert scheduler.get_window_average() is None  # Window not full yet

    def test_window_full(self, optimizer: SGD):
        """Test window average when window is full."""
        scheduler = WindowedReduceLROnPlateau(optimizer, window_size=5, update_every=1)

        for i in range(5):
            scheduler.step(float(i))  # 0, 1, 2, 3, 4

        assert len(scheduler._loss_window) == 5
        assert scheduler.get_window_average() == 2.0  # (0+1+2+3+4)/5

    def test_window_sliding(self, optimizer: SGD):
        """Test that window slides (old values pushed out)."""
        scheduler = WindowedReduceLROnPlateau(optimizer, window_size=3, update_every=1)

        for i in range(5):
            scheduler.step(float(i))  # Window should contain [2, 3, 4]

        assert list(scheduler._loss_window) == [2.0, 3.0, 4.0]
        assert scheduler.get_window_average() == 3.0

    def test_update_every_skips_updates(self, optimizer: SGD):
        """Test that scheduler only updates every N steps."""
        scheduler = WindowedReduceLROnPlateau(optimizer, window_size=2, update_every=3, patience=0, factor=0.5)

        # Fill window with high loss
        scheduler.step(10.0)
        scheduler.step(10.0)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Step 3 should trigger update (window full, step_count=3)
        scheduler.step(10.0)
        # But patience=0 means it needs one more "bad" update to reduce

        # Steps 4, 5 - no update
        scheduler.step(10.0)
        scheduler.step(10.0)
        assert optimizer.param_groups[0]["lr"] == initial_lr  # No change yet

        # Step 6 should trigger update
        scheduler.step(10.0)
        # Now we should see LR reduction after patience is exhausted

    def test_lr_reduction_on_plateau(self, optimizer: SGD):
        """Test that LR is reduced when loss plateaus."""
        scheduler = WindowedReduceLROnPlateau(
            optimizer,
            window_size=2,
            update_every=1,
            patience=2,
            factor=0.5,
            threshold=0.0,
        )
        initial_lr = optimizer.param_groups[0]["lr"]

        # Fill window and trigger updates with constant loss
        for _ in range(10):
            scheduler.step(1.0)

        # After patience exhausted, LR should be reduced
        assert optimizer.param_groups[0]["lr"] < initial_lr

    def test_lr_no_reduction_when_improving(self, optimizer: SGD):
        """Test that LR is not reduced when loss is improving."""
        scheduler = WindowedReduceLROnPlateau(
            optimizer,
            window_size=2,
            update_every=1,
            patience=2,
            factor=0.5,
        )
        initial_lr = optimizer.param_groups[0]["lr"]

        # Continuously improving loss
        for i in range(10, 0, -1):
            scheduler.step(float(i))

        # LR should not be reduced
        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_state_dict_save_load(self, optimizer: SGD):
        """Test that state can be saved and loaded."""
        scheduler = WindowedReduceLROnPlateau(optimizer, window_size=5, update_every=10)

        # Add some state
        for i in range(3):
            scheduler.step(float(i))

        state = scheduler.state_dict()
        assert state["window_size"] == 5
        assert state["update_every"] == 10
        assert state["loss_window"] == [0.0, 1.0, 2.0]
        assert state["step_count"] == 3

        # Create new scheduler and load state
        new_scheduler = WindowedReduceLROnPlateau(optimizer, window_size=1, update_every=1)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.window_size == 5
        assert new_scheduler.update_every == 10
        assert list(new_scheduler._loss_window) == [0.0, 1.0, 2.0]
        assert new_scheduler._step_count == 3

    def test_mode_max(self, optimizer: SGD):
        """Test scheduler works with mode='max'."""
        scheduler = WindowedReduceLROnPlateau(
            optimizer,
            window_size=2,
            update_every=1,
            patience=2,
            factor=0.5,
            mode="max",
            threshold=0.0,
        )
        initial_lr = optimizer.param_groups[0]["lr"]

        # Constant low metric (bad for max mode)
        for _ in range(10):
            scheduler.step(0.1)

        # LR should be reduced
        assert optimizer.param_groups[0]["lr"] < initial_lr

    def test_cooldown(self, optimizer: SGD):
        """Test that cooldown delays subsequent reductions."""
        scheduler = WindowedReduceLROnPlateau(
            optimizer,
            window_size=2,
            update_every=1,
            patience=1,
            factor=0.5,
            cooldown=10,
            threshold=0.0,
        )
        initial_lr = optimizer.param_groups[0]["lr"]

        # Trigger first reduction (need patience+1 bad updates after window fills)
        for _ in range(5):
            scheduler.step(1.0)

        # Should have had one reduction by now
        lr_after_first_reduction = optimizer.param_groups[0]["lr"]
        assert lr_after_first_reduction < initial_lr

        # Record how many reductions happened with cooldown
        reduction_count_with_cooldown = 0
        for _ in range(15):
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(1.0)
            if optimizer.param_groups[0]["lr"] < old_lr:
                reduction_count_with_cooldown += 1

        # Now test without cooldown - should reduce more frequently
        optimizer2 = SGD(torch.nn.Linear(10, 1).parameters(), lr=0.1)
        scheduler2 = WindowedReduceLROnPlateau(
            optimizer2,
            window_size=2,
            update_every=1,
            patience=1,
            factor=0.5,
            cooldown=0,
            threshold=0.0,
        )

        # Same warmup
        for _ in range(5):
            scheduler2.step(1.0)

        reduction_count_without_cooldown = 0
        for _ in range(15):
            old_lr = optimizer2.param_groups[0]["lr"]
            scheduler2.step(1.0)
            if optimizer2.param_groups[0]["lr"] < old_lr:
                reduction_count_without_cooldown += 1

        # With cooldown, should have fewer reductions
        assert reduction_count_with_cooldown < reduction_count_without_cooldown

    def test_min_lr(self, optimizer: SGD):
        """Test that LR does not go below min_lr."""
        scheduler = WindowedReduceLROnPlateau(
            optimizer,
            window_size=2,
            update_every=1,
            patience=1,
            factor=0.1,
            min_lr=0.01,
            threshold=0.0,
        )

        # Many updates with constant loss to trigger multiple reductions
        for _ in range(100):
            scheduler.step(1.0)

        assert optimizer.param_groups[0]["lr"] >= 0.01
