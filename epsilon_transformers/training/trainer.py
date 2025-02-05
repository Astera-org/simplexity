import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
import time
import numpy as np
from pathlib import Path

from epsilon_transformers.training.logger import StructuredLogger
from epsilon_transformers.training.models import RNNWrapper
from epsilon_transformers.training.configs import TrainConfig

class Trainer:
    """Main trainer class that handles both Transformer and RNN training"""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Set precision
        torch.set_float32_matmul_precision('high')
        
        # Initialize components
        self._init_model()
        self._init_optimizer()
        self._init_scheduler()
        self._init_logger()
        self._init_persister()
        
        # Prepare evaluation data
        self._prepare_eval_data()

    def _init_model(self):
        """Initialize model based on config type"""
        if self.config.model.type == "transformer":
            self.model = self.config.model.to_hooked_transformer(
                device=self.device.type, 
                seed=self.config.seed
            )
        else:  # RNN
            rnn = torch.nn.RNN(
                input_size=self.config.model.d_vocab,
                hidden_size=self.config.model.d_model,
                num_layers=self.config.model.n_layers,
                nonlinearity='relu'
            )
            output_layer = torch.nn.Linear(
                self.config.model.d_model,
                self.config.model.d_vocab
            )
            self.model = RNNWrapper(rnn, output_layer).to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = self.config.optimizer.from_model(
            model=self.model, 
            device=self.device
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=200,
            cooldown=500,
            threshold=1e-6,
            verbose=True
        )

    def _init_logger(self):
        """Initialize training logger"""
        self.logger = StructuredLogger(
            Path(self.config.persistance.collection_location) / "train_log.csv"
        )

    def _init_persister(self):
        """Initialize model persister"""
        self.persister = self.config.persistance.init()
        self.persister.save_config(self.config)

    def _prepare_eval_data(self):
        """Prepare data for evaluation"""
        msp_tree = self.config.dataset.process_msp(10)
        self.eval_probs, self.beliefs, self.eval_sequences = (
            msp_tree.collect_paths_with_beliefs(max_depth=9)
        )
        self.myopic_entropy = msp_tree.myopic_entropy[1:]

    def train_epoch(self) -> np.ndarray:
        """Train for one epoch and return losses"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        epoch_losses = []
        dataset = self.config.dataset.to_dataloader_gpu(
            self.config.model.n_ctx, 
            True
        )

        for batch in dataset:
            # Prepare input and target sequences
            input_sequences = batch[:, :-1]
            targets = batch[:, 1:].reshape(-1)
            
            # Forward pass
            logits = self.model(input_sequences)
            reshaped_logits = logits.reshape(-1, self.model.cfg.d_vocab)
            
            # Compute loss
            loss = F.cross_entropy(
                reshaped_logits,
                targets,
                reduction="none"
            ).reshape(batch.shape[0], -1)
            
            # Backward pass
            loss.mean().backward()
            epoch_losses.append(loss)

        # Optimizer step
        self.optimizer.step()
        
        # Return mean loss across all batches
        return torch.concat(epoch_losses).mean(axis=0).detach().cpu().numpy()

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Run model evaluation"""
        self.model.eval()
        with torch.no_grad():
            # TODO: Implement evaluation logic
            # This should return relative_loss, kl_div, belief_predictions
            pass

    def train(self, num_epochs: int = 1001, save_every: int = 50):
        """Main training loop"""
        for epoch in range(num_epochs):
            start_time = time.monotonic_ns()
            
            # Train epoch
            loss = self.train_epoch()
            
            # Calculate metrics
            time_ms = (time.monotonic_ns() - start_time) / 1e6
            rel_loss = self.myopic_entropy / loss[:len(self.myopic_entropy)]
            
            # Log training metrics
            self.logger.log_train(
                epoch, 
                loss, 
                rel_loss, 
                iter_time_ms=time_ms,
                lr=self.optimizer.param_groups[0]['lr']
            )
            
            # Update learning rate
            self.scheduler.step(loss.mean())
            
            # Periodic evaluation and saving
            if epoch % save_every == 0 and epoch != 0:
                relative_loss, kl_div, belief_predictions = self.evaluate()
                
                # Create visualization if needed
                figure = None
                if self.config.dataset.process == "mess3":
                    figure = self._plot_beliefs(belief_predictions)
                
                # Log validation metrics
                self.logger.log_valid(
                    epoch, 
                    loss, 
                    relative_loss, 
                    kl_div, 
                    figure=figure
                )
                
                # Save model
                self.persister.save_model(
                    self.model,
                    epoch * self.config.dataset.batch_size * self.config.dataset.num_tokens
                )

    def _plot_beliefs(self, belief_predictions: torch.Tensor):
        """Create visualization of belief predictions"""
        # TODO: Implement visualization logic
        pass