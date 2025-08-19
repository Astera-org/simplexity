# Neural network components for transformer model
"""Neural network components including transformer blocks, attention, and embeddings."""

from typing import Any

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HookPoint",
    "Embed",
    "Unembed",
    "PosEmbed",
    "LayerNorm",
    "Attention",
    "MLP",
    "TransformerBlock",
    "Transformer",
]


class HookPoint(nn.Module):
    """A helper module for accessing intermediate activations.

    Acts as identity by default, but allows easy addition of PyTorch hooks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fwd_hooks: list[Any] = []
        self.bwd_hooks: list[Any] = []
        self.name: str = ""

    def give_name(self, name: str) -> None:
        """Set the name of this hook point."""
        self.name = name

    def add_hook(self, hook: Any, direction: str = "fwd") -> None:
        """Add a hook function. Hook format: fn(activation, hook_name)."""

        def full_hook(module: nn.Module, module_input: Any, module_output: Any) -> Any:
            return hook(module_output, name=self.name)

        if direction == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif direction == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'fwd' or 'bwd'")

    def remove_hooks(self, direction: str = "fwd") -> None:
        """Remove hooks from this hook point."""
        if direction in ["fwd", "both"]:
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []

        if direction in ["bwd", "both"]:
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []

        if direction not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction: {direction}")

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the hook point."""
        return x


class Embed(nn.Module):
    """Token embedding layer."""

    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the embedding layer."""
        return t.einsum("dbp -> bpd", self.W_E[:, x])


class Unembed(nn.Module):
    """Unembedding layer to convert hidden states to logits."""

    def __init__(self, d_vocab: int, d_model: int) -> None:
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the unembedding layer."""
        return x @ self.W_U


class PosEmbed(nn.Module):
    """Positional embedding layer."""

    def __init__(self, max_ctx: int, d_model: int) -> None:
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the positional embedding."""
        return x + self.W_pos[: x.shape[-2]]


class LayerNorm(nn.Module):
    """Layer normalization with optional disable."""

    def __init__(self, d_model: int, epsilon: float = 1e-4, use_ln: bool = False) -> None:
        super().__init__()
        self.use_ln = use_ln
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the layer normalization."""
        if not self.use_ln:
            return x

        x_centered = x - x.mean(dim=-1, keepdim=True)
        x_normalized = x_centered / (x_centered.std(dim=-1, keepdim=True) + self.epsilon)
        return x_normalized * self.w_ln + self.b_ln


class Attention(nn.Module):
    """Multi-head attention with causal masking."""

    def __init__(self, d_model: int, num_heads: int, d_head: int, n_ctx: int) -> None:
        super().__init__()
        self.d_head = d_head

        # Weight matrices
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))

        # Causal mask
        self.register_buffer("mask", t.tril(t.ones((n_ctx, n_ctx))))

        # Hook points for analysis
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the attention layer."""
        # Compute key, query, value
        k = self.hook_k(t.einsum("ihd,bpd->biph", self.W_K, x))
        q = self.hook_q(t.einsum("ihd,bpd->biph", self.W_Q, x))
        v = self.hook_v(t.einsum("ihd,bpd->biph", self.W_V, x))

        # Attention scores
        attn_scores = t.einsum("biph,biqh->biqp", k, q)

        # Apply causal mask
        seq_len = x.shape[-2]
        mask = self.mask[:seq_len, :seq_len]  # type: ignore[index]
        masked_scores = t.tril(attn_scores) - 1e10 * (1 - mask)

        # Softmax attention
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(masked_scores / np.sqrt(self.d_head)), dim=-1))

        # Apply attention to values
        z = self.hook_z(t.einsum("biph,biqp->biqh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")

        # Output projection
        return t.einsum("df,bqf->bqd", self.W_O, z_flat)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation."""

    def __init__(self, d_model: int, d_mlp: int, act_type: str) -> None:
        super().__init__()
        if act_type not in ["ReLU", "GeLU"]:
            raise ValueError(f"Unknown activation type: {act_type}")

        self.W_in = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))

        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the MLP."""
        # Input projection
        x = self.hook_pre(t.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)

        # Activation
        if self.act_type == "ReLU":
            x = F.relu(x)
        elif self.act_type == "GeLU":
            x = F.gelu(x)

        x = self.hook_post(x)

        # Output projection
        return t.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out


class TransformerBlock(nn.Module):
    """A single transformer block with attention and MLP."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        n_ctx: int,
        act_type: str,
    ) -> None:
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp, act_type)

        # Hook points for residual stream
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the transformer block."""
        # Attention with residual connection
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.hook_resid_pre(x))))

        # MLP with residual connection
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))

        return x


class Transformer(nn.Module):
    """Main transformer model for modular arithmetic."""

    def __init__(
        self,
        d_vocab: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_head: int,
        d_mlp: int,
        n_ctx: int,
        act_type: str,
        use_cache: bool = False,
    ) -> None:
        """Initialize the transformer model."""
        super().__init__()
        self.cache: dict[str, Any] = {}
        self.use_cache = use_cache

        # Model components
        self.embed = Embed(d_vocab=d_vocab, d_model=d_model)
        self.pos_embed = PosEmbed(max_ctx=n_ctx, d_model=d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    num_heads=num_heads,
                    n_ctx=n_ctx,
                    act_type=act_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.unembed = Unembed(d_vocab=d_vocab, d_model=d_model)

        # Initialize hook point names
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass of the transformer model."""
        # Embedding and positional encoding
        x = self.embed(x)
        x = self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Unembedding to logits
        return self.unembed(x)

    def set_use_cache(self, use_cache: bool) -> None:
        """Enable or disable caching."""
        self.use_cache = use_cache

    def hook_points(self) -> list[HookPoint]:
        """Get all hook points in the model."""
        return [module for _, module in self.named_modules() if isinstance(module, HookPoint)]

    def remove_all_hooks(self) -> None:
        """Remove all hooks from the model."""
        for hp in self.hook_points():
            hp.remove_hooks("both")

    def cache_all(self, cache: dict[str, Any], include_backward: bool = False) -> None:
        """Cache all activations wrapped in HookPoints."""

        def save_hook(tensor: t.Tensor, name: str) -> None:
            cache[name] = tensor.detach()

        def save_hook_back(tensor: t.Tensor, name: str) -> None:
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if include_backward:
                hp.add_hook(save_hook_back, "bwd")


def build_pytorch_transformer(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_mlp: int,
    n_ctx: int,
    act_type: str,
    use_ln: bool = False,
) -> Transformer:
    """Build a PyTorch transformer model from configuration parameters.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model embeddings
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_mlp: Dimension of the MLP hidden layer
        n_ctx: Maximum context length
        act_type: Activation function type ("ReLU" or "GeLU")
        use_ln: Whether to use layer normalization

    Returns:
        Configured PyTorch transformer model
    """
    # Calculate d_head from d_model and num_heads
    d_head = d_model // num_heads

    return Transformer(
        d_vocab=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        n_ctx=n_ctx,
        act_type=act_type,
        use_cache=False,
    )
