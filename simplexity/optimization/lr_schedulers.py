"""Custom learning rate schedulers with windowed averaging."""

from collections import deque
from typing import Any, Literal

import hydra
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class WindowedReduceLROnPlateau(ReduceLROnPlateau):
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
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
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

    def step(self, metrics: float, epoch: int | None = None) -> None:  # type: ignore[override]
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


class LinearWarmupScheduler(LRScheduler):
    """Learning rate scheduler with linear warmup.

    Linearly increases the learning rate from `warmup_start_factor * base_lr` to
    `base_lr` over `warmup_steps` steps. After warmup, optionally delegates to a
    wrapped scheduler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for the warmup phase.
        warmup_start_factor: Starting factor for learning rate during warmup. Default: 0.01.
        wrapped_scheduler_cfg: Optional Hydra config dict for wrapped scheduler. Default: None.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_factor: float = 0.01,
        wrapped_scheduler_cfg: dict[str, Any] | DictConfig | None = None,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self._warmup_step_count = 0

        if wrapped_scheduler_cfg is not None:
            self.wrapped_scheduler: LRScheduler | None = hydra.utils.instantiate(
                wrapped_scheduler_cfg, optimizer=optimizer
            )
        else:
            self.wrapped_scheduler = None

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current step."""
        if self._warmup_step_count < self.warmup_steps:
            alpha = self._warmup_step_count / self.warmup_steps
            factor = self.warmup_start_factor + alpha * (1.0 - self.warmup_start_factor)
            return [base_lr * factor for base_lr in self.base_lrs]
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metrics: float | None = None, epoch: int | None = None) -> None:  # type: ignore[override]
        """Take a step and update the learning rate.

        During warmup, linearly interpolates the learning rate. After warmup,
        delegates to the wrapped scheduler if configured.

        Args:
            metrics: Optional metric value for ReduceLROnPlateau-style schedulers.
            epoch: Optional epoch number (passed to wrapped scheduler).
        """
        self._warmup_step_count += 1
        if self._warmup_step_count <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(), strict=False):
                param_group["lr"] = lr
        elif self.wrapped_scheduler is not None:
            if isinstance(self.wrapped_scheduler, ReduceLROnPlateau):
                self.wrapped_scheduler.step(metrics, epoch)
            else:
                self.wrapped_scheduler.step(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state including warmup state and wrapped scheduler."""
        state: dict[str, Any] = {
            "warmup_steps": self.warmup_steps,
            "warmup_start_factor": self.warmup_start_factor,
            "warmup_step_count": self._warmup_step_count,
            "base_lrs": self.base_lrs,
        }
        if self.wrapped_scheduler is not None:
            state["wrapped_scheduler_state"] = self.wrapped_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state including warmup state and wrapped scheduler."""
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.warmup_start_factor = state_dict.get("warmup_start_factor", self.warmup_start_factor)
        self._warmup_step_count = state_dict.get("warmup_step_count", 0)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
        if self.wrapped_scheduler is not None and "wrapped_scheduler_state" in state_dict:
            self.wrapped_scheduler.load_state_dict(state_dict["wrapped_scheduler_state"])
