"""TransformerLens model wrapper for simplexity integration."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError as e:
    raise ImportError(
        "TransformerLens is required for this module.\n"
        "Install it with: pip install transformer-lens"
    ) from e


class TransformerLensWrapper(nn.Module):
    """Wrapper to make TransformerLens models compatible with simplexity's PyTorch training."""
    
    def __init__(
        self,
        d_model: int = 64,
        d_head: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        n_ctx: int = 64,
        d_vocab: int = 3,
        act_fn: str = "relu",
        normalization_type: str = "LN",
        device: Optional[str] = None,
        seed: int = 42,
        attn_only: bool = False,
        use_cache: bool = True,
        use_hook_tokens: bool = True,
    ):
        """Initialize TransformerLens model with configuration.
        
        Args:
            d_model: Model dimension (embedding size)
            d_head: Dimension of each attention head
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            n_ctx: Context window (max sequence length)
            d_vocab: Vocabulary size (3 for mess3)
            act_fn: Activation function for MLPs
            normalization_type: Type of normalization ("LN" for LayerNorm)
            device: Device to use (None for auto-detection)
            seed: Random seed for initialization
            attn_only: If True, only use attention (no MLPs)
            use_cache: Enable caching for interpretability
            use_hook_tokens: Enable hook points for analysis
        """
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create HookedTransformerConfig
        self.config = HookedTransformerConfig(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_layers=n_layers,
            n_ctx=n_ctx,
            d_vocab=d_vocab,
            act_fn=act_fn,
            normalization_type=normalization_type,
            device=device,
            seed=seed,
            attn_only=attn_only,
            use_cache=use_cache,
            use_hook_tokens=use_hook_tokens,
        )
        
        # Create the HookedTransformer model
        self.model = HookedTransformer(self.config)
        self.device = device
        
    def forward(self, input_ids: torch.Tensor, return_loss: bool = True) -> torch.Tensor:
        """Forward pass compatible with simplexity's training.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            return_loss: If True, compute and return the loss
            
        Returns:
            If return_loss is True, returns scalar loss
            Otherwise, returns logits of shape (batch_size, sequence_length, vocab_size)
        """
        if return_loss:
            # For training, compute loss directly
            # TransformerLens expects the labels to be the same as inputs (next token prediction)
            loss = self.model(input_ids, return_type="loss")
            return loss
        else:
            # For inference, return logits
            logits = self.model(input_ids, return_type="logits")
            return logits
    
    def run_with_cache(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Run model with caching for interpretability analysis.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Tuple of (logits, cache_dict) where cache_dict contains all activations
        """
        return self.model.run_with_cache(input_ids)
    
    def get_attention_patterns(self, input_ids: torch.Tensor, layer: int = 0) -> torch.Tensor:
        """Extract attention patterns for a given layer.
        
        Args:
            input_ids: Input token IDs
            layer: Which layer's attention to extract
            
        Returns:
            Attention patterns of shape (batch, heads, seq_len, seq_len)
        """
        _, cache = self.run_with_cache(input_ids)
        return cache[f"pattern", layer]
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self


def create_transformerlens_model(
    vocab_size: int,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    device: Optional[str] = None,
    **kwargs
) -> TransformerLensWrapper:
    """Create a TransformerLens model for simplexity.
    
    Args:
        vocab_size: Size of vocabulary (3 for mess3)
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        device: Device to use
        **kwargs: Additional configuration parameters
        
    Returns:
        TransformerLensWrapper instance
    """
    return TransformerLensWrapper(
        d_vocab=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        device=device,
        **kwargs
    )