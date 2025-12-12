"""Custom learning rate schedulers with windowed averaging."""

from collections import deque
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WindowedReduceLROnPlateau(ReduceLROnPlateau):  # pylint: disable=abstract-method
    """ReduceLROnPlateau with windowed average loss comparison.

    Instead of comparing individual loss values, this scheduler compares the
    average loss over a sliding window. This smooths out noise in batch losses
    and makes the patience mechanism more effective.

    Losses are accumulated every time `step()` is called. The underlying
    ReduceLROnPlateau is only updated every `update_every` calls (once the
    window has filled), using the average of the last `window_size` losses.

    Args:
        optimizer: Wrapped optimizer.
        window_size: Number of recent losses to average. Default: 10.
        update_every: Only update the scheduler every N steps. Default: 1.
        mode: One of "min" or "max". Default: "min".
        factor: Factor by which the learning rate will be reduced. Default: 0.1.
        patience: Number of updates with no improvement after which LR is reduced. Default: 10.
        threshold: Threshold for measuring the new optimum. Default: 1e-4.
        threshold_mode: One of "rel" or "abs". Default: "rel".
        cooldown: Number of updates to wait before resuming normal operation. Default: 0.
        min_lr: Minimum learning rate. Default: 0.
        eps: Minimal decay applied to lr. Default: 1e-8.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        optimizer: Optimizer,
        window_size: int = 10,
        update_every: int = 1,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | list[float] = 0,
        eps: float = 1e-8,
    ):
        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        self.window_size = window_size
        self.update_every = update_every
        self._loss_window: deque[float] = deque(maxlen=window_size)
        self._step_count = 0

    def step(self, metrics: float | Any, epoch: int | None = None) -> None:  # type: ignore[override]
        """Record a loss value and potentially update LR based on windowed average.

        Losses are accumulated every call. The underlying scheduler is only
        updated every `update_every` calls once the window is full.

        Args:
            metrics: Current loss value to add to the window.
            epoch: Optional epoch number (passed to parent).
        """
        current = float(metrics)
        self._loss_window.append(current)
        self._step_count += 1

        if len(self._loss_window) < self.window_size:
            return

        if self._step_count % self.update_every != 0:
            return

        avg_loss = sum(self._loss_window) / len(self._loss_window)
        super().step(avg_loss, epoch)

    def get_window_average(self) -> float | None:
        """Return the current window average, or None if window not full."""
        if len(self._loss_window) < self.window_size:
            return None
        return sum(self._loss_window) / len(self._loss_window)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state including the loss window."""
        state = super().state_dict()
        state["window_size"] = self.window_size
        state["update_every"] = self.update_every
        state["loss_window"] = list(self._loss_window)
        state["step_count"] = self._step_count
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state including the loss window."""
        self.window_size = state_dict.pop("window_size", self.window_size)
        self.update_every = state_dict.pop("update_every", self.update_every)
        loss_window = state_dict.pop("loss_window", [])
        self._step_count = state_dict.pop("step_count", 0)
        self._loss_window = deque(loss_window, maxlen=self.window_size)
        super().load_state_dict(state_dict)
