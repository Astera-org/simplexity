from __future__ import annotations

from omegaconf import OmegaConf

from simplexity.utils.pytorch_utils import resolve_device


def resolve_model_device(resolved_cfg) -> str:
    """Mutate resolved Hydra config to use the best available device.

    Sets `resolved_cfg.model.instance.cfg.device` to a concrete, supported
    device string for the current platform (cuda > mps > cpu), regardless of
    what the original run used. Returns the chosen device string.
    """
    try:
        # Default to 'auto' if field missing; always normalize to concrete device
        desired = getattr(getattr(getattr(resolved_cfg, "model", None), "instance", None), "cfg", None)
        current = getattr(desired, "device", "auto") if desired is not None else "auto"
        chosen = resolve_device(str(current) if current is not None else "auto")

        was_struct = OmegaConf.is_struct(resolved_cfg)
        OmegaConf.set_struct(resolved_cfg, False)
        try:
            if not hasattr(resolved_cfg, "model"):
                resolved_cfg.model = OmegaConf.create({})
            if not hasattr(resolved_cfg.model, "instance"):
                resolved_cfg.model.instance = OmegaConf.create({})
            if not hasattr(resolved_cfg.model.instance, "cfg"):
                resolved_cfg.model.instance.cfg = OmegaConf.create({})
            resolved_cfg.model.instance.cfg.device = chosen
        finally:
            OmegaConf.set_struct(resolved_cfg, was_struct)
        return chosen
    except Exception:
        # Best-effort: if anything goes wrong, do not crash analysis/training
        return "cpu"
