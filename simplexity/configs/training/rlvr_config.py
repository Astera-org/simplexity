"""Configuration for RLVR training using TRL."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RLVRConfig:
    """Configuration for RLVR training process."""
    
    # Basic training parameters
    seed: int
    num_epochs: int
    samples_per_epoch: int
    max_prompt_length: int
    max_generation_length: int
    complexity_range: Tuple[int, int]
    
    # PPO-specific parameters
    ppo_steps: int
    learning_rate: float
    batch_size: int
    mini_batch_size: int
    gradient_accumulation_steps: int
    ppo_epochs: int
    cliprange: float
    cliprange_value: float
    vf_coef: float
    target_kl: float
    early_stopping: bool
    
    # Generation parameters
    temperature: float
    top_p: float
    
    # Reward parameters
    reward_type: str  # "boxed", "correct", or "combined"
    boxed_weight: float
    correct_weight: float
    
    # Logging and checkpointing
    log_every: Optional[int]
    checkpoint_every: Optional[int]
    max_batches_per_epoch: int


def validate_rlvr_config(cfg: RLVRConfig) -> None:
    """Validate the RLVR configuration."""
    assert cfg.num_epochs > 0, "Number of epochs must be greater than 0"
    assert cfg.samples_per_epoch > 0, "Samples per epoch must be greater than 0"
    assert cfg.max_prompt_length > 0, "Max prompt length must be greater than 0"
    assert cfg.max_generation_length > cfg.max_prompt_length, "Max generation length must be greater than prompt length"
    assert cfg.complexity_range[0] >= 1, "Minimum complexity must be at least 1"
    assert cfg.complexity_range[1] >= cfg.complexity_range[0], "Max complexity must be >= min complexity"
    
    # PPO parameter validation
    assert cfg.ppo_steps > 0, "PPO steps must be greater than 0"
    assert cfg.learning_rate > 0, "Learning rate must be greater than 0"
    assert cfg.batch_size > 0, "Batch size must be greater than 0"
    assert cfg.mini_batch_size > 0, "Mini batch size must be greater than 0"
    assert cfg.mini_batch_size <= cfg.batch_size, "Mini batch size must be <= batch size"
    assert cfg.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"
    assert cfg.ppo_epochs > 0, "PPO epochs must be greater than 0"
    assert 0 < cfg.cliprange <= 1, "Cliprange must be between 0 and 1"
    assert 0 < cfg.cliprange_value <= 1, "Cliprange value must be between 0 and 1"
    assert cfg.vf_coef >= 0, "Value function coefficient must be non-negative"
    assert cfg.target_kl > 0, "Target KL must be greater than 0"
    
    # Generation parameter validation
    assert cfg.temperature > 0, "Temperature must be greater than 0"
    assert 0 < cfg.top_p <= 1, "Top-p must be between 0 and 1"
    
    # Reward parameter validation
    assert cfg.reward_type in ["boxed", "correct", "combined"], f"Invalid reward type: {cfg.reward_type}"
    assert cfg.boxed_weight >= 0, "Boxed weight must be non-negative"
    assert cfg.correct_weight >= 0, "Correct weight must be non-negative"
    assert cfg.boxed_weight + cfg.correct_weight > 0, "At least one reward weight must be positive"
    
    # Logging validation
    if cfg.log_every is not None:
        assert cfg.log_every > 0, "Log every must be greater than 0"
    if cfg.checkpoint_every is not None:
        assert cfg.checkpoint_every > 0, "Checkpoint every must be greater than 0"
    
    assert cfg.max_batches_per_epoch > 0, "Max batches per epoch must be greater than 0"