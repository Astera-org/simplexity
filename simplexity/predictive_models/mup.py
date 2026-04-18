"""muP (Maximal Update Parameterization) integration for HookedTransformer.

Applies three post-instantiation mutations required for muP-correct training:

1. Rescales attention pre-softmax by `1/d_head` instead of the default `1/sqrt(d_head)`.
2. Divides the unembed output by `width_mult = d_model / base_d_model`, matching
   the behaviour of `mup.MuReadout` while preserving the `unembed.W_U` attribute
   that `transformer_lens.HookedTransformer` references in many places.
3. Records `.infshape` metadata on every parameter via `mup.set_base_shapes`,
   optionally rescaling initial weights. This metadata is consumed by
   `mup.MuAdam` / `mup.MuAdamW` to scale per-parameter learning rates.

The base and delta "shape models" are cheap `HookedTransformer` instances built
at reduced widths solely for shape inference.

Call order matters: apply muP *before* loading a checkpoint. `set_base_shapes`
rewrites parameter data when `rescale_params=True`, so loading first and
applying second would overwrite the loaded values. Loading a muP-trained
checkpoint requires a muP-applied target model so that `.infshape` metadata is
present for `MuAdam.step()`.

Note on `rescale_params`: the upstream `mup` library only rescales parameters
living inside `nn.Linear`, `mup.MuReadout`, or `_ConvNd` modules. `transformer_lens`
uses raw `nn.Parameter` tensors inside custom modules (`Attention`, `MLP`,
`Unembed`), so `rescale_params=True` is effectively a no-op for
`HookedTransformer`. muP correctness is still achieved via `.infshape` +
`MuAdam` (per-parameter LR scaling), the unembed forward patch (readout output
scaling), and the attention-scale patch (1/d attention).
"""

from __future__ import annotations

from typing import Any

from simplexity.logger import SIMPLEXITY_LOGGER


def _import_mup() -> Any:
    """Import `mup` with a helpful error message if the extra is missing."""
    try:
        import mup
    except ImportError as exc:
        raise ImportError("muP support requires the `mup` package. Install with `uv sync --extra mup`.") from exc
    return mup


def _import_transformer_lens() -> tuple[Any, Any]:
    """Import `transformer_lens.HookedTransformer` and its config class."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    return HookedTransformer, HookedTransformerConfig


def default_overrides(
    real_cfg: Any,
    base_d_model: int,
    delta_d_model: int,
    base_d_head: int | None = None,
    base_d_mlp: int | None = None,
    delta_d_head: int | None = None,
    delta_d_mlp: int | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute consistent base and delta config overrides from a real HookedTransformer config.

    Fills in `d_head` and `d_mlp` from `d_model` when not provided so the shape
    models stay internally consistent.

    Args:
        real_cfg: The real `HookedTransformerConfig` (or compatible DictConfig).
        base_d_model: Target `d_model` for the base shape model.
        delta_d_model: Target `d_model` for the delta shape model.
        base_d_head: Optional base `d_head`; default `base_d_model // n_heads`.
        base_d_mlp: Optional base `d_mlp`; default `4 * base_d_model`.
        delta_d_head: Optional delta `d_head`; default `delta_d_model // n_heads`.
        delta_d_mlp: Optional delta `d_mlp`; default `4 * delta_d_model`.

    Returns:
        A pair of override dicts `(base_overrides, delta_overrides)`.
    """
    n_heads = real_cfg.n_heads
    if n_heads is None or n_heads <= 0:
        n_heads = real_cfg.d_model // real_cfg.d_head

    base_overrides = {
        "d_model": base_d_model,
        "d_head": base_d_head if base_d_head is not None else max(base_d_model // n_heads, 1),
        "d_mlp": base_d_mlp if base_d_mlp is not None else 4 * base_d_model,
    }
    delta_overrides = {
        "d_model": delta_d_model,
        "d_head": delta_d_head if delta_d_head is not None else max(delta_d_model // n_heads, 1),
        "d_mlp": delta_d_mlp if delta_d_mlp is not None else 4 * delta_d_model,
    }
    return base_overrides, delta_overrides


def _build_shape_model(real_cfg: Any, overrides: dict[str, int]) -> Any:
    """Build a HookedTransformer with overridden widths for shape inference."""
    hooked_transformer_cls, hooked_transformer_config_cls = _import_transformer_lens()
    cfg_kwargs = dict(real_cfg.to_dict()) if hasattr(real_cfg, "to_dict") else dict(real_cfg.__dict__)
    for key, value in overrides.items():
        cfg_kwargs[key] = value
    cfg_kwargs["device"] = "cpu"
    cfg_kwargs["n_heads"] = cfg_kwargs["d_model"] // cfg_kwargs["d_head"]
    shape_cfg = hooked_transformer_config_cls.from_dict(cfg_kwargs)
    return hooked_transformer_cls(shape_cfg)


def _patch_attention_scale(model: Any) -> None:
    """Override attention scale to 1/d_head on every block."""
    d_head = float(model.cfg.d_head)
    for block in model.blocks:
        block.attn.attn_scale = d_head


def _patch_unembed_forward(model: Any, width_mult: float) -> None:
    """Rescale the unembed output by 1/width_mult at forward time."""
    unembed = model.unembed
    if getattr(unembed, "_mup_patched", False):
        unembed._mup_width_mult = width_mult
        return
    original_forward = unembed.forward
    unembed._mup_width_mult = width_mult

    def _mup_forward(residual, _orig=original_forward, _mod=unembed):
        return _orig(residual) / _mod._mup_width_mult

    unembed.forward = _mup_forward
    unembed._mup_patched = True


def apply_mup_to_hooked_transformer(
    model: Any,
    base_cfg_overrides: dict[str, int],
    delta_cfg_overrides: dict[str, int],
    rescale_params: bool = True,
) -> Any:
    """Apply muP to a `HookedTransformer` in place and return it.

    Args:
        model: A `transformer_lens.HookedTransformer` instance.
        base_cfg_overrides: Width overrides for the base shape model
            (keys: `d_model`, `d_head`, `d_mlp`).
        delta_cfg_overrides: Width overrides for the delta shape model.
        rescale_params: Whether `mup.set_base_shapes` should rescale initialized
            weights. Must be `True` for fresh-init training, `False` when
            followed by a checkpoint load of a previously muP-rescaled model.

    Returns:
        The same `model`, now with patched attention scale, patched unembed
        forward, and `.infshape` metadata on every parameter.

    Raises:
        ImportError: If the `mup` package is not installed.
    """
    mup = _import_mup()

    real_d_model = model.cfg.d_model
    base_d_model = base_cfg_overrides["d_model"]
    width_mult = real_d_model / base_d_model

    base_model = _build_shape_model(model.cfg, base_cfg_overrides)
    delta_model = _build_shape_model(model.cfg, delta_cfg_overrides)

    _patch_attention_scale(model)
    _patch_attention_scale(base_model)
    _patch_attention_scale(delta_model)

    base_width_mult = base_d_model / base_d_model
    delta_width_mult = delta_cfg_overrides["d_model"] / base_d_model
    _patch_unembed_forward(model, width_mult)
    _patch_unembed_forward(base_model, base_width_mult)
    _patch_unembed_forward(delta_model, delta_width_mult)

    mup.set_base_shapes(model, base_model, delta=delta_model, rescale_params=rescale_params)

    SIMPLEXITY_LOGGER.info(
        "[muP] applied (real_d_model=%d, base_d_model=%d, delta_d_model=%d, width_mult=%.3f)",
        real_d_model,
        base_d_model,
        delta_cfg_overrides["d_model"],
        width_mult,
    )
    return model
