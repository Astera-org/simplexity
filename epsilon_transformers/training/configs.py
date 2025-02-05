from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, Dict, Any
import torch
from transformer_lens import HookedTransformer

class OptimizerConfig(BaseModel):
    """Configuration for optimizer"""
    type: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    def from_model(self, model: torch.nn.Module, device: str) -> torch.optim.Optimizer:
        """Create optimizer instance from config"""
        optimizer_dict = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        optimizer_cls = optimizer_dict[self.type.lower()]
        return optimizer_cls(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

class TransformerConfig(BaseModel):
    """Configuration for transformer model"""
    type: Literal["transformer"] = "transformer"
    d_model: int = 256
    d_head: int = 16
    n_layers: int = 10
    n_ctx: int = 10
    n_heads: int = 4
    d_mlp: Optional[int] = None  # If None, will be set to 4*d_model
    d_vocab: int = 2
    act_fn: str = "relu"
    attn_only: bool = False
    normalization_type: str = "LN"

    def to_hooked_transformer(self, device: str, seed: int) -> "HookedTransformer":
        """Create HookedTransformer instance from config"""
        from transformer_lens import HookedTransformer, HookedTransformerConfig
        
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model

        config = HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            n_ctx=self.n_ctx,
            n_heads=self.n_heads,
            d_mlp=self.d_mlp,
            d_vocab=self.d_vocab,
            seed=seed,
            device=device,
            act_fn=self.act_fn,
            attn_only=self.attn_only,
            normalization_type=self.normalization_type,
        )
        return HookedTransformer(config)

class RNNConfig(BaseModel):
    """Configuration for RNN model"""
    type: Literal["rnn"] = "rnn"
    d_model: int = 256  # hidden size
    n_layers: int = 1
    n_ctx: int = 10
    d_vocab: int = 2
    nonlinearity: Literal["tanh", "relu"] = "relu"

class DatasetConfig(BaseModel):
    """Configuration for dataset"""
    process: str = "mess3"  # e.g., "mess3", "rrxor"
    process_params: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 128
    num_tokens: int = 65238
    fixed_train_dataset: bool = False

    def to_dataloader_gpu(self, n_ctx: int, train: bool = True):
        """Create GPU DataLoader from config"""
        # TODO: Implement dataset creation logic
        pass

    def process_msp(self, depth: int):
        """Create MSP tree from process config"""
        # TODO: Implement MSP tree creation
        pass

class TrainConfig(BaseModel):
    """Main training configuration"""
    # Basic training settings
    seed: int = 42
    wandb_project: Optional[str] = None
    
    # Model configuration
    model: Union[TransformerConfig, RNNConfig]
    
    # Training components
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    
    @classmethod
    def from_args(cls, args: Any) -> "TrainConfig":
        """Create config from command line arguments"""
        # Create model config based on model type
        if args.model == "transformer":
            model_config = TransformerConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                # ... other transformer-specific args
            )
        else:  # RNN
            model_config = RNNConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                # ... other RNN-specific args
            )

        return cls(
            seed=args.seed,
            wandb_project=args.wandb_project,
            model=model_config,
            optimizer=OptimizerConfig(
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
            ),
            dataset=DatasetConfig(
                process=args.process,
                batch_size=args.batch_size,
                # ... other dataset args
            )
        )