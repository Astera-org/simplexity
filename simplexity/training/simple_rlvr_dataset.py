"""Simplified PyTorch-only dataset for RLVR training."""

from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset
import random


class SimpleArithmeticDataset(Dataset):
    """Simplified arithmetic dataset that doesn't rely on JAX generation.
    
    This creates simple arithmetic problems directly in PyTorch.
    """
    
    def __init__(
        self,
        tokens: Dict[str, int],
        p: int,
        num_samples: int,
        max_prompt_length: int,
        seed: int = 42
    ):
        """Initialize the simple arithmetic dataset.
        
        Args:
            tokens: Token dictionary from arithmetic process
            p: Modulus for arithmetic operations
            num_samples: Number of samples to generate
            max_prompt_length: Maximum prompt length
            seed: Random seed
        """
        self.tokens = tokens
        self.p = p
        self.num_samples = num_samples
        self.max_prompt_length = max_prompt_length
        
        # Extract token IDs
        self.boe_token = tokens["<boe>"]
        self.eql_token = tokens["="]
        self.eoe_token = tokens["<eoe>"]
        self.pad_token = tokens["<pad>"]
        self.add_token = tokens["+"]
        
        random.seed(seed)
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate arithmetic samples."""
        self.samples = []
        
        for _ in range(self.num_samples):
            # Generate simple addition problems: a + b = ?
            a = random.randint(0, min(self.p - 1, 12))
            b = random.randint(0, min(self.p - 1, 12))
            
            # Create prompt: <boe> a b + = 
            prompt = [self.boe_token, a, b, self.add_token, self.eql_token]
            
            # Pad to max length
            while len(prompt) < self.max_prompt_length:
                prompt.append(self.pad_token)
            
            # Truncate if too long
            prompt = prompt[:self.max_prompt_length]
            
            # Store the correct answer for reference
            correct_answer = (a + b) % self.p
            
            self.samples.append({
                "prompt": prompt,
                "correct_answer": correct_answer,
                "operands": (a, b),
            })
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        sample = self.samples[idx]
        
        prompt_tensor = torch.tensor(sample["prompt"], dtype=torch.long)
        attention_mask = (prompt_tensor != self.pad_token).long()
        
        return {
            "input_ids": prompt_tensor,
            "attention_mask": attention_mask,
            "correct_answer": torch.tensor(sample["correct_answer"], dtype=torch.long),
            "operands": torch.tensor(sample["operands"], dtype=torch.long),
        }


def create_simple_rlvr_trainer(model, tokens, p, config):
    """Create a simplified RLVR trainer that doesn't rely on complex TRL integration."""
    
    class SimpleRLVRTrainer:
        def __init__(self, model, tokens, p, config):
            self.model = model
            self.tokens = tokens
            self.p = p
            self.config = config
            
            # Setup device
            self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            self.model = self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
            
            # Setup dataset
            self.dataset = SimpleArithmeticDataset(
                tokens=tokens,
                p=p,
                num_samples=config.get("samples_per_epoch", 100),
                max_prompt_length=config.get("max_prompt_length", 20),
                seed=config.get("seed", 42),
            )
            
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=config.get("batch_size", 4),
                shuffle=True,
            )
            
            # Reward calculator
            from simplexity.training.reward_functions import ArithmeticRewardCalculator
            self.reward_calculator = ArithmeticRewardCalculator(tokens, p)
        
        def generate_sequence(self, prompt: torch.Tensor, max_new_tokens: int = 10) -> torch.Tensor:
            """Generate a sequence from a prompt."""
            generated_tokens = []
            current_seq = prompt.clone()
            
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(current_seq)
                    next_token_logits = logits[0, -1, :]
                    
                    # Apply temperature
                    temperature = self.config.get("temperature", 1.0)
                    next_token_logits = next_token_logits / temperature
                    
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated_tokens.append(next_token.item())
                    current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop at end of equation
                    if next_token.item() == self.tokens["<eoe>"]:
                        break
            
            return torch.tensor(generated_tokens, dtype=torch.long, device=self.device)
        
        def train_epoch(self) -> Dict[str, float]:
            """Train for one epoch."""
            total_loss = 0.0
            total_reward = 0.0
            num_batches = 0
            
            for batch in self.dataloader:
                prompts = batch["input_ids"].to(self.device)
                correct_answers = batch["correct_answer"].to(self.device)
                batch_size = prompts.shape[0]
                
                # Generate sequences and compute rewards
                batch_loss = 0.0
                batch_reward = 0.0
                
                for i in range(batch_size):
                    prompt = prompts[i:i+1]
                    
                    # Generate with gradient tracking
                    generated = []
                    log_probs = []
                    current_seq = prompt.clone()
                    
                    for step in range(10):  # Max 10 new tokens
                        logits = self.model(current_seq)
                        next_token_logits = logits[0, -1, :]
                        
                        # Apply temperature
                        next_token_logits = next_token_logits / self.config.get("temperature", 1.0)
                        
                        # Sample with gradient tracking
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token_dist = torch.distributions.Categorical(probs)
                        next_token = next_token_dist.sample()
                        log_prob = next_token_dist.log_prob(next_token)
                        
                        generated.append(next_token.item())
                        log_probs.append(log_prob)
                        
                        # Add to sequence
                        current_seq = torch.cat([current_seq, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        
                        # Stop at end of equation
                        if next_token.item() == self.tokens["<eoe>"]:
                            break
                    
                    # Compute reward
                    full_sequence = current_seq.squeeze(0)
                    reward = self.reward_calculator.boxed_answer_reward(full_sequence.unsqueeze(0))
                    reward_value = float(reward[0])
                    
                    # Compute policy gradient loss
                    if log_probs:
                        total_log_prob = torch.stack(log_probs).sum()
                        loss = -total_log_prob * reward_value
                        batch_loss += loss
                        batch_reward += reward_value
                
                # Backpropagation
                if batch_loss != 0:
                    avg_loss = batch_loss / batch_size
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += float(avg_loss.item())
                    total_reward += batch_reward / batch_size
                    num_batches += 1
                
                # Early stopping for demo
                if num_batches >= self.config.get("max_batches_per_epoch", 5):
                    break
            
            metrics = {}
            if num_batches > 0:
                metrics["loss"] = total_loss / num_batches
                metrics["reward"] = total_reward / num_batches
            
            return metrics
        
        def train(self, num_epochs: int):
            """Train the model."""
            for epoch in range(num_epochs):
                metrics = self.train_epoch()
                print(f"Epoch {epoch + 1}: Loss={metrics.get('loss', 0):.4f}, Reward={metrics.get('reward', 0):.4f}")
            
            return self.model
    
    return SimpleRLVRTrainer(model, tokens, p, config)