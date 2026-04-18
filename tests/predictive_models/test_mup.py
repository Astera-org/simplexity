"""Tests for muP (Maximal Update Parameterization) integration with HookedTransformer."""

from __future__ import annotations

import math

import pytest

mup = pytest.importorskip("mup", reason="muP tests require the `mup` optional extra")
torch = pytest.importorskip("torch", reason="muP tests require torch")
transformer_lens = pytest.importorskip("transformer_lens", reason="muP tests require transformer_lens")

from simplexity.predictive_models.mup import (  # noqa: E402
    apply_mup_to_hooked_transformer,
    default_overrides,
)


def _build_model(d_model: int = 16, seed: int = 0):
    cfg = transformer_lens.HookedTransformerConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model // 4,
        n_heads=4,
        n_ctx=8,
        d_mlp=4 * d_model,
        d_vocab=4,
        act_fn="relu",
        normalization_type="LN",
        device="cpu",
        seed=seed,
    )
    return transformer_lens.HookedTransformer(cfg)


def _apply(model, rescale_params: bool = True):
    base_overrides, delta_overrides = default_overrides(model.cfg, base_d_model=8, delta_d_model=24)
    return apply_mup_to_hooked_transformer(model, base_overrides, delta_overrides, rescale_params=rescale_params)


class TestApplyMuP:
    """Unit tests for apply_mup_to_hooked_transformer."""

    def test_sets_infshape_on_parameters(self):
        model = _build_model()
        _apply(model)
        assert hasattr(model.unembed.W_U, "infshape")
        assert hasattr(model.blocks[0].attn.W_Q, "infshape")
        assert hasattr(model.embed.W_E, "infshape")

    def test_patches_attn_scale_to_d_head(self):
        model = _build_model(d_model=16)
        default_scale = model.blocks[0].attn.attn_scale
        assert default_scale == pytest.approx(math.sqrt(model.cfg.d_head))
        _apply(model)
        assert model.blocks[0].attn.attn_scale == pytest.approx(float(model.cfg.d_head))

    def test_patches_unembed_forward_scales_by_inverse_width_mult(self):
        model = _build_model(d_model=16)
        _apply(model, rescale_params=False)
        assert getattr(model.unembed, "_mup_patched", False) is True
        assert model.unembed._mup_width_mult == pytest.approx(16 / 8)

    def test_rescale_params_flag_accepted(self):
        """Both rescale_params values must run without error and set infshape.

        transformer_lens uses raw `nn.Parameter` objects rather than `nn.Linear`,
        so `rescale_params=True` in `mup.set_base_shapes` is effectively a no-op
        for this architecture; muP correctness is carried by `.infshape`
        metadata (consumed by `MuAdam`), the unembed forward patch, and the
        attention-scale patch. This test guards the flag plumbing only.
        """
        for rescale in (True, False):
            model = _build_model(d_model=16, seed=42)
            _apply(model, rescale_params=rescale)
            assert hasattr(model.blocks[0].attn.W_Q, "infshape")

    def test_mu_adamw_constructs(self):
        model = _build_model()
        _apply(model)
        opt = mup.MuAdamW(model.parameters(), lr=1e-3)
        assert opt is not None
        assert len(opt.param_groups) > 1

    def test_forward_still_runs(self):
        model = _build_model().to("cpu")
        _apply(model)
        tokens = torch.randint(0, 4, (2, 8), device="cpu")
        out = model(tokens)
        assert out.shape == (2, 8, 4)

    def test_single_training_step(self):
        model = _build_model().to("cpu")
        _apply(model)
        opt = mup.MuAdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, 4, (2, 8), device="cpu")
        logits = model(tokens)
        loss = logits.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()


class TestDefaultOverrides:
    """Unit tests for default_overrides helper."""

    def test_derives_d_head_and_d_mlp_from_d_model(self):
        model = _build_model(d_model=16)
        base_overrides, delta_overrides = default_overrides(model.cfg, base_d_model=8, delta_d_model=32)
        assert base_overrides == {"d_model": 8, "d_head": 2, "d_mlp": 32}
        assert delta_overrides == {"d_model": 32, "d_head": 8, "d_mlp": 128}

    def test_respects_explicit_overrides(self):
        model = _build_model(d_model=16)
        base_overrides, delta_overrides = default_overrides(
            model.cfg,
            base_d_model=8,
            delta_d_model=32,
            base_d_head=4,
            base_d_mlp=16,
            delta_d_head=16,
            delta_d_mlp=256,
        )
        assert base_overrides == {"d_model": 8, "d_head": 4, "d_mlp": 16}
        assert delta_overrides == {"d_model": 32, "d_head": 16, "d_mlp": 256}
