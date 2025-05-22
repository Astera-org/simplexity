from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the validation."""

    seed: int
    sequence_len: int
    batch_size: int
    num_steps: int
    log_every: int | None


def validate_config(cfg: Config) -> None:
    """Validate the configuration."""
    assert cfg.sequence_len > 0, "Sequence length must be greater than 0"
    assert cfg.batch_size > 0, "Batch size must be greater than 0"
    assert cfg.num_steps > 0, "Number of steps must be greater than 0"
    if cfg.log_every is not None:
        assert cfg.log_every > 0, "Log every must be greater than 0"
        assert cfg.log_every <= cfg.num_steps, "Log every must be less than or equal to number of steps"
