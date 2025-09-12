"""Reward functions for RLVR training of transformer models on arithmetic tasks."""

from typing import Dict, Any
import torch
import torch.nn.functional as F


class ArithmeticRewardCalculator:
    """Calculator for arithmetic task rewards compatible with TRL training."""
    
    def __init__(self, tokens: Dict[str, int], p: int):
        """Initialize the reward calculator.
        
        Args:
            tokens: Dictionary mapping token strings to token IDs
            p: Modulus for arithmetic operations (determines valid operand range)
        """
        self.tokens = tokens
        self.p = p
        
        # Extract special token IDs
        self.eql_token = tokens["="]
        self.eoe_token = tokens["<eoe>"]
        self.boe_token = tokens["<boe>"]
        self.pad_token = tokens["<pad>"]
    
    def boxed_answer_reward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute boxed answer reward for sequences.
        
        Rewards sequences where the <EOE> token is immediately preceded by 
        the <EQL> token and an operand token.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len) containing token IDs
            
        Returns:
            Tensor of shape (batch_size,) with reward values (0.0 or 1.0)
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Find positions of EOE tokens for each sequence
        eoe_mask = (sequences == self.eoe_token)
        
        # Get the position of the last EOE token in each sequence
        # If no EOE token exists, this will be 0 (which is fine for our logic)
        eoe_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            eoe_indices = torch.where(eoe_mask[i])[0]
            if len(eoe_indices) > 0:
                eoe_positions[i] = eoe_indices[-1]
        
        # Check if there are at least 2 tokens before EOE for EQL and operand
        valid_position = (eoe_positions >= 2)
        
        # Check if token at eoe_pos - 2 is EQL token
        eql_positions = torch.clamp(eoe_positions - 2, 0, seq_len - 1)
        correct_eql = (sequences[torch.arange(batch_size), eql_positions] == self.eql_token)
        
        # Check if token at eoe_pos - 1 is an operand (value < p)
        operand_positions = torch.clamp(eoe_positions - 1, 0, seq_len - 1)
        is_operand = (sequences[torch.arange(batch_size), operand_positions] < self.p)
        
        # Combine all conditions
        reward = (valid_position & correct_eql & is_operand).float()
        
        return reward
    
    def correct_answer_reward(self, sequences: torch.Tensor, correct_answers: torch.Tensor) -> torch.Tensor:
        """Compute correct answer reward for sequences.
        
        Rewards sequences where the <EOE> token is immediately preceded by 
        the <EQL> token and the correct answer.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len) containing token IDs
            correct_answers: Tensor of shape (batch_size,) with correct answer tokens
            
        Returns:
            Tensor of shape (batch_size,) with reward values (0.0 or 1.0)
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Find positions of EOE tokens for each sequence
        eoe_mask = (sequences == self.eoe_token)
        
        # Get the position of the last EOE token in each sequence
        eoe_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            eoe_indices = torch.where(eoe_mask[i])[0]
            if len(eoe_indices) > 0:
                eoe_positions[i] = eoe_indices[-1]
        
        # Check if there are at least 2 tokens before EOE for EQL and answer
        valid_position = (eoe_positions >= 2)
        
        # Check if token at eoe_pos - 2 is EQL token
        eql_positions = torch.clamp(eoe_positions - 2, 0, seq_len - 1)
        correct_eql = (sequences[torch.arange(batch_size), eql_positions] == self.eql_token)
        
        # Check if token at eoe_pos - 1 matches the correct answer
        answer_positions = torch.clamp(eoe_positions - 1, 0, seq_len - 1)
        correct_answer_match = (sequences[torch.arange(batch_size), answer_positions] == correct_answers)
        
        # Combine all conditions
        reward = (valid_position & correct_eql & correct_answer_match).float()
        
        return reward
    
    def combined_reward(self, sequences: torch.Tensor, correct_answers: torch.Tensor, 
                       boxed_weight: float = 0.3, correct_weight: float = 0.7) -> torch.Tensor:
        """Compute a combined reward that weights both boxed and correct answer rewards.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len) containing token IDs
            correct_answers: Tensor of shape (batch_size,) with correct answer tokens
            boxed_weight: Weight for the boxed answer reward
            correct_weight: Weight for the correct answer reward
            
        Returns:
            Tensor of shape (batch_size,) with combined reward values
        """
        boxed_rewards = self.boxed_answer_reward(sequences)
        correct_rewards = self.correct_answer_reward(sequences, correct_answers)
        
        # Combined reward: both rewards must be satisfied for full points
        # But partial credit given for just having correct format
        combined = boxed_weight * boxed_rewards + correct_weight * (boxed_rewards * correct_rewards)
        
        return combined


def create_reward_function(tokens: Dict[str, int], p: int, reward_type: str = "combined"):
    """Factory function to create reward functions for TRL training.
    
    Args:
        tokens: Dictionary mapping token strings to token IDs
        p: Modulus for arithmetic operations
        reward_type: Type of reward ("boxed", "correct", or "combined")
        
    Returns:
        Callable reward function compatible with TRL
    """
    calculator = ArithmeticRewardCalculator(tokens, p)
    
    if reward_type == "boxed":
        def reward_fn(sequences, **kwargs):
            return calculator.boxed_answer_reward(sequences)
    elif reward_type == "correct":
        def reward_fn(sequences, correct_answers, **kwargs):
            return calculator.correct_answer_reward(sequences, correct_answers)
    elif reward_type == "combined":
        def reward_fn(sequences, correct_answers, **kwargs):
            return calculator.combined_reward(sequences, correct_answers)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward_fn