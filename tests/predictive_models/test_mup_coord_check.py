"""Coordinate-check test for muP on HookedTransformer.

The muP coordinate check is the standard validation that per-coordinate
activation magnitudes stay bounded as width grows. Under standard
parameterization (SP), activations at a fixed hook grow with width; under
muP they stay roughly constant. This test trains a few steps at multiple
widths with a fixed seed and asserts the relative activation scale does not
blow up as width doubles.
"""

from __future__ import annotations

import pytest

mup = pytest.importorskip("mup", reason="muP coord check requires the `mup` optional extra")
torch = pytest.importorskip("torch", reason="muP coord check requires torch")
transformer_lens = pytest.importorskip("transformer_lens", reason="muP coord check requires transformer_lens")

from simplexity.predictive_models.mup import (  # noqa: E402
    apply_mup_to_hooked_transformer,
    default_overrides,
)

HOOK_POINTS = ("hook_embed", "blocks.0.hook_resid_post", "blocks.0.attn.hook_z")
BASE_D_MODEL = 16
DELTA_D_MODEL = 48
WIDTHS = (32, 64, 128)
NUM_STEPS = 3
BATCH_SIZE = 4
SEQ_LEN = 8
VOCAB_SIZE = 8


def _build_muP_model(d_model: int, seed: int):
    n_heads = 4
    cfg = transformer_lens.HookedTransformerConfig(
        n_layers=2,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_ctx=SEQ_LEN,
        d_mlp=4 * d_model,
        d_vocab=VOCAB_SIZE,
        act_fn="relu",
        normalization_type="LN",
        device="cpu",
        seed=seed,
    )
    model = transformer_lens.HookedTransformer(cfg)
    base_overrides, delta_overrides = default_overrides(
        model.cfg, base_d_model=BASE_D_MODEL, delta_d_model=DELTA_D_MODEL
    )
    return apply_mup_to_hooked_transformer(model, base_overrides, delta_overrides, rescale_params=True)


def _train_and_measure(d_model: int, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    model = _build_muP_model(d_model, seed).to("cpu")
    optimizer = mup.MuAdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device="cpu")
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device="cpu")

    for _ in range(NUM_STEPS):
        logits, cache = model.run_with_cache(tokens, names_filter=list(HOOK_POINTS))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, final_cache = model.run_with_cache(tokens, names_filter=list(HOOK_POINTS))
    return {hp: final_cache[hp].detach().abs().mean().item() for hp in HOOK_POINTS}


@pytest.mark.slow
def test_mup_coord_check_activation_scale_bounded():
    """After a few optimizer steps, per-coordinate activation magnitudes at each
    tracked hookpoint should not scale with width under muP. Tolerance is loose
    (max(width)/min(width) * 1.5) to accommodate stochasticity and the small
    step budget, while still catching standard-param-style linear blow-up.
    """
    measurements = {width: _train_and_measure(width, seed=0) for width in WIDTHS}

    for hook in HOOK_POINTS:
        values = [measurements[w][hook] for w in WIDTHS]
        ratio = max(values) / max(min(values), 1e-8)
        width_ratio = max(WIDTHS) / min(WIDTHS)
        assert ratio < 1.5 * width_ratio, (
            f"Coord check failed at hook {hook}: activation ratio {ratio:.2f} across widths "
            f"{WIDTHS} exceeds threshold. Values: {dict(zip(WIDTHS, values, strict=True))}"
        )
