"""RLVR training using TRL (Transformer Reinforcement Learning) library."""

import warnings
from typing import Optional, Dict, Any, List, Tuple
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel, PretrainedConfig
except ImportError as e:
    raise ImportError(
        "To use RLVR training, install TRL and dependencies:\n"
        "pip install trl transformers accelerate\n"
        "Or: pip install --break-system-packages trl transformers accelerate"
    ) from e

import jax
import jax.numpy as jnp

from simplexity.configs.training.rlvr_config import RLVRConfig
from simplexity.generative_processes.arithmetic_process import ArithmeticProcess
from simplexity.logging.logger import Logger
from simplexity.training.reward_functions import ArithmeticRewardCalculator
from simplexity.training.rlvr_dataset import ArithmeticPromptDataset


class SimpleTransformerWrapper:
    """Simple wrapper for the transformer model to handle generation."""
    
    def __init__(self, model: nn.Module, vocab_size: int, pad_token_id: int = 0):
        """Initialize wrapper.
        
        Args:
            model: The custom transformer model
            vocab_size: Size of the vocabulary
            pad_token_id: ID of the padding token
        """
        self.model = model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass."""
        return self.model(input_ids)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_p: float = 1.0, **kwargs):
        """Generate sequences."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                logits = self.model(generated)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                if top_p < 1.0:
                    # Simplified top-p implementation
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumsum_probs = torch.cumsum(probs, dim=-1)
                    
                    # Create mask for top-p
                    mask = cumsum_probs > top_p
                    next_token_logits.scatter_(-1, sorted_indices, mask.float() * (-float('inf')))
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if max length reached
                if generated.shape[1] >= max_length:
                    break
        
        return generated


class ArithmeticRLVRTrainer:
    """RLVR trainer for arithmetic tasks using TRL."""
    
    def __init__(
        self,
        model: nn.Module,
        arithmetic_process: ArithmeticProcess,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
    ):
        """Initialize the RLVR trainer.
        
        Args:
            model: The transformer model to train
            arithmetic_process: Process for generating arithmetic data
            config: Configuration dictionary for training
            logger: Optional logger for metrics
        """
        self.model = model
        self.arithmetic_process = arithmetic_process
        self.config = config
        self.logger = logger
        
        # Setup device
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        
        # Wrap model for generation
        self.wrapped_model = SimpleTransformerWrapper(
            model, 
            vocab_size=arithmetic_process.vocab_size,
            pad_token_id=arithmetic_process.tokens["<pad>"]
        )
        
        # Setup reward calculator
        self.reward_calculator = ArithmeticRewardCalculator(
            tokens=arithmetic_process.tokens,
            p=arithmetic_process.p
        )
        
        # For now, use a simplified approach without full TRL integration
        # We'll implement a custom PPO-like training loop
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-5))
        
        # Store other training parameters
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.cliprange = config.get("cliprange", 0.2)
        self.target_kl = config.get("target_kl", 0.1)
        
        # Setup dataset
        self.dataset = ArithmeticPromptDataset(
            arithmetic_process=arithmetic_process,
            num_samples=config.get("samples_per_epoch", 1000),
            max_prompt_length=config.get("max_prompt_length", 50),
            complexity_range=config.get("complexity_range", (1, 3)),
            seed=config.get("seed", 42),
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.ppo_config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for the dataloader."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        complexity = torch.tensor([item["complexity"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "complexity": complexity,
        }
    

    
    def _evaluate_arithmetic_expression(self, sequence: jnp.ndarray) -> int:
        """Evaluate arithmetic expression to get the correct answer.
        
        This is a simplified version that tries to extract the initial expression
        and evaluate it using the arithmetic process.
        
        Args:
            sequence: Token sequence
            
        Returns:
            Correct answer token
        """
        # Find the beginning of equation and first equals
        boe_token = self.arithmetic_process.tokens["<boe>"]
        eql_token = self.arithmetic_process.tokens["="]
        
        boe_pos = jnp.where(sequence == boe_token)[0]
        eql_pos = jnp.where(sequence == eql_token)[0]
        
        if len(boe_pos) > 0 and len(eql_pos) > 0:
            start = int(boe_pos[0]) + 1
            end = int(eql_pos[0])
            
            if end > start:
                sub_expr = sequence[start:end]
                # Use the arithmetic process to evaluate this
                # This is a simplified approach
                try:
                    if hasattr(self.arithmetic_process, 'child_sub_equation'):
                        n = len(sub_expr)
                        _, evaluated = self.arithmetic_process.child_sub_equation(sub_expr)
                        # Find the final result
                        non_pad = evaluated != self.arithmetic_process.tokens["<pad>"]
                        if jnp.any(non_pad):
                            result_candidates = evaluated[non_pad]
                            # Take the last non-padding token as the result
                            return int(result_candidates[-1])
                except:
                    pass
        
        # Fallback: return a random operand
        return 0
    
    def _generate_with_log_probs(self, prompt: torch.Tensor, max_new_tokens: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens with log probability tracking.
        
        Args:
            prompt: Input prompt tensor of shape (1, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (generated_tokens, log_probs) where generated_tokens is the
            new tokens only (without prompt) and log_probs are the log probabilities
        """
        generated_tokens = []
        log_probs = []
        
        current_seq = prompt.clone()
        
        for _ in range(max_new_tokens):
            # Get logits from model
            with torch.enable_grad():
                logits = self.model(current_seq)
                next_token_logits = logits[0, -1, :]  # Last position, remove batch dim
                
                # Apply temperature
                next_token_logits = next_token_logits / self.temperature
                
                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(probs, dim=-1)
                
                # Find the cutoff index for top-p
                cutoff_idx = torch.where(cumsum_probs > self.top_p)[0]
                if len(cutoff_idx) > 0:
                    cutoff_idx = cutoff_idx[0]
                    # Zero out probabilities beyond cutoff
                    next_token_logits[sorted_indices[cutoff_idx:]] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Track log probability
                log_prob = torch.log(probs[next_token] + 1e-8)
                log_probs.append(log_prob)
                generated_tokens.append(next_token)
                
                # Add token to sequence for next iteration
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                
                # Stop at end of equation or padding
                if next_token.item() == self.arithmetic_process.tokens["<eoe>"]:
                    break
        
        if generated_tokens:
            generated_tensor = torch.cat(generated_tokens, dim=0)
            log_probs_tensor = torch.cat(log_probs, dim=0)
        else:
            generated_tensor = torch.tensor([], dtype=torch.long, device=prompt.device)
            log_probs_tensor = torch.tensor([], dtype=torch.float32, device=prompt.device)
        
        return generated_tensor, log_probs_tensor
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step using policy gradients."""
        metrics = {}
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            prompts = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            batch_size = prompts.shape[0]
            
            # Generate sequences with the model
            generated_sequences = []
            log_probs_list = []
            
            for b in range(batch_size):
                prompt = prompts[b:b+1]  # Keep batch dimension
                
                # Simple generation with tracking of log probabilities
                generated, log_probs = self._generate_with_log_probs(prompt)
                generated_sequences.append(generated)
                log_probs_list.append(log_probs)
            
            # Compute rewards for generated sequences
            rewards = []
            for i, generated in enumerate(generated_sequences):
                # Create full sequence (prompt + generated)
                prompt_seq = prompts[i]
                
                # Remove padding from prompt
                prompt_no_pad = prompt_seq[prompt_seq != self.arithmetic_process.tokens["<pad>"]]
                
                # Combine prompt and generated
                full_seq = torch.cat([prompt_no_pad, generated])
                
                # Truncate to reasonable length and compute reward
                max_len = min(len(full_seq), self.config.get("max_generation_length", 100))
                truncated_seq = full_seq[:max_len]
                
                reward = self.reward_calculator.boxed_answer_reward(truncated_seq.unsqueeze(0))
                rewards.append(float(reward[0]))
            
            # Convert rewards to tensor
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # Compute policy gradient loss
            policy_loss = 0.0
            for i, log_probs in enumerate(log_probs_list):
                # Simple REINFORCE: loss = -log_prob * reward
                reward = rewards_tensor[i]
                policy_loss += -log_probs.sum() * reward
            
            policy_loss = policy_loss / batch_size
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += float(policy_loss.item())
            total_reward += float(rewards_tensor.mean().item())
            num_batches += 1
            
            # Early stopping for debugging
            if batch_idx >= self.config.get("max_batches_per_epoch", 10):
                break
        
        # Average metrics
        if num_batches > 0:
            metrics["policy_loss"] = total_loss / num_batches
            metrics["reward_mean"] = total_reward / num_batches
        
        return metrics
    
    def train(self, num_epochs: int) -> nn.Module:
        """Train the model using RLVR.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Trained model
        """
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Perform training step
            metrics = self.train_step()
            
            # Log metrics
            if self.logger:
                epoch_metrics = {f"rlvr/{k}": v for k, v in metrics.items()}
                self.logger.log_metrics(epoch + 1, epoch_metrics)
            
            # Print progress
            if metrics:
                reward_mean = metrics.get("reward_mean", 0.0)
                print(f"Epoch {epoch + 1} - Average Reward: {reward_mean:.4f}")
        
        return self.model


def train_rlvr(
    model: nn.Module,
    rlvr_cfg: Any,
    arithmetic_process: ArithmeticProcess,
    logger: Optional[Logger] = None,
    **kwargs
) -> tuple[nn.Module, float]:
    """Train a model using RLVR with TRL.
    
    Args:
        model: The transformer model to train
        rlvr_cfg: RLVR training configuration
        arithmetic_process: Arithmetic process for data generation
        logger: Optional logger
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (trained_model, final_reward)
    """
    # Convert RLVR config to dictionary for RLVR trainer
    rlvr_config = {
        "batch_size": getattr(rlvr_cfg, "batch_size", 8),
        "learning_rate": getattr(rlvr_cfg, "learning_rate", 1e-5),
        "seed": getattr(rlvr_cfg, "seed", 42),
        "samples_per_epoch": getattr(rlvr_cfg, "samples_per_epoch", 1000),
        "max_prompt_length": getattr(rlvr_cfg, "max_prompt_length", 25),
        "max_generation_length": getattr(rlvr_cfg, "max_generation_length", 50),
        "complexity_range": getattr(rlvr_cfg, "complexity_range", (1, 3)),
        "temperature": getattr(rlvr_cfg, "temperature", 0.7),
        "top_p": getattr(rlvr_cfg, "top_p", 0.9),
        "ppo_steps": getattr(rlvr_cfg, "ppo_steps", 100),
        "max_batches_per_epoch": getattr(rlvr_cfg, "max_batches_per_epoch", 10),
    }
    
    # Initialize trainer
    trainer = ArithmeticRLVRTrainer(
        model=model,
        arithmetic_process=arithmetic_process,
        config=rlvr_config,
        logger=logger,
    )
    
    # Train
    num_epochs = getattr(rlvr_cfg, "num_epochs", 10)
    trained_model = trainer.train(num_epochs)
    
    # Return final reward as loss (for compatibility)
    final_reward = 0.0  # This would be computed from final evaluation
    
    return trained_model, final_reward