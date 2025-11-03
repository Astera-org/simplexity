def is_optimizer_target(target: str) -> bool:
    """Check if the target is a optimizer target."""
    return target.startswith("torch.optim.") or target.startswith("optax.")
