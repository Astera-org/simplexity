"""Dataset classes for RLVR training with TRL."""

from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import jax
import jax.numpy as jnp

from simplexity.generative_processes.arithmetic_process import ArithmeticProcess


class ArithmeticRLVRDataset(Dataset):
    """Dataset for RLVR training on arithmetic tasks.
    
    This dataset generates arithmetic equations and provides prompts for the model
    to complete, along with the correct answers for reward calculation.
    """
    
    def __init__(
        self,
        arithmetic_process: ArithmeticProcess,
        num_samples: int,
        sequence_length: int,
        complexity: int,
        prompt_length_ratio: float = 0.7,
        seed: int = 42
    ):
        """Initialize the RLVR dataset.
        
        Args:
            arithmetic_process: The arithmetic process for generating equations
            num_samples: Number of samples to generate
            sequence_length: Maximum sequence length
            complexity: Complexity parameter for equation generation
            prompt_length_ratio: Ratio of sequence to use as prompt (rest is for completion)
            seed: Random seed for reproducibility
        """
        self.arithmetic_process = arithmetic_process
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.complexity = complexity
        self.prompt_length = int(sequence_length * prompt_length_ratio)
        self.seed = seed
        
        # Generate all samples upfront for reproducibility
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate all samples for the dataset."""
        np.random.seed(self.seed)
        jax_key = jax.random.PRNGKey(self.seed)
        
        self.samples = []
        self.correct_answers = []
        
        for i in range(self.num_samples):
            # Generate a complete equation
            key = jax.random.fold_in(jax_key, i)
            _, equation = self.arithmetic_process.generate(
                self.complexity, key, self.sequence_length, False
            )
            
            # Convert to numpy for easier manipulation
            equation_np = np.array(equation)
            
            # Extract the correct answer (token before EOE)
            eoe_token = self.arithmetic_process.tokens["<eoe>"]
            eoe_positions = np.where(equation_np == eoe_token)[0]
            
            if len(eoe_positions) > 0:
                eoe_pos = eoe_positions[-1]
                if eoe_pos >= 1:
                    correct_answer = equation_np[eoe_pos - 1]
                else:
                    correct_answer = 0  # Fallback
            else:
                correct_answer = 0  # Fallback
            
            # Create prompt by truncating the equation
            # Find a good truncation point (after an operator or operand, before the final answer)
            prompt = self._create_prompt(equation_np)
            
            self.samples.append(prompt)
            self.correct_answers.append(correct_answer)
    
    def _create_prompt(self, equation: np.ndarray) -> np.ndarray:
        """Create a prompt from a complete equation.
        
        The prompt should end at a point where the model needs to complete
        the arithmetic reasoning, typically after the initial expression
        but before the final evaluation steps.
        
        Args:
            equation: Complete equation array
            
        Returns:
            Prompt array (truncated equation)
        """
        # Find the first equals sign - this marks the start of the evaluation
        eql_token = self.arithmetic_process.tokens["="]
        eql_positions = np.where(equation == eql_token)[0]
        
        if len(eql_positions) > 0:
            # Truncate just before the first equals sign
            # This means the model needs to evaluate the expression
            first_eql = eql_positions[0]
            prompt_end = min(first_eql, self.prompt_length)
        else:
            # Fallback to fixed length
            prompt_end = self.prompt_length
        
        # Ensure we don't truncate too early
        prompt_end = max(prompt_end, 10)  # Minimum prompt length
        
        # Create prompt and pad if necessary
        prompt = equation[:prompt_end]
        
        # Pad to consistent length for batching
        if len(prompt) < self.prompt_length:
            pad_token = self.arithmetic_process.tokens["<pad>"]
            padding = np.full(self.prompt_length - len(prompt), pad_token)
            prompt = np.concatenate([prompt, padding])
        
        return prompt
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing prompt, correct_answer, and metadata
        """
        prompt = torch.tensor(self.samples[idx], dtype=torch.long)
        correct_answer = torch.tensor(self.correct_answers[idx], dtype=torch.long)
        
        return {
            "input_ids": prompt,
            "correct_answer": correct_answer,
            "complexity": self.complexity,
        }


class ArithmeticPromptDataset(Dataset):
    """Simplified dataset that only provides prompts for TRL training.
    
    This is more suitable for online generation during training.
    """
    
    def __init__(
        self,
        arithmetic_process: ArithmeticProcess,
        num_samples: int,
        max_prompt_length: int,
        complexity_range: Tuple[int, int] = (1, 3),
        seed: int = 42
    ):
        """Initialize the prompt dataset.
        
        Args:
            arithmetic_process: The arithmetic process for generating equations
            num_samples: Number of samples per epoch
            max_prompt_length: Maximum length of prompts
            complexity_range: Range of complexity values to sample from
            seed: Random seed
        """
        self.arithmetic_process = arithmetic_process
        self.num_samples = num_samples
        self.max_prompt_length = max_prompt_length
        self.complexity_range = complexity_range
        self.seed = seed
        
        # Tokens for prompt creation
        self.boe_token = arithmetic_process.tokens["<boe>"]
        self.eql_token = arithmetic_process.tokens["="]
        self.pad_token = arithmetic_process.tokens["<pad>"]
    
    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a prompt sample.
        
        Args:
            idx: Sample index (used as seed modifier)
            
        Returns:
            Dictionary with input_ids (prompt) and metadata
        """
        # Generate a complexity level for this sample
        np.random.seed(self.seed + idx)
        complexity = np.random.randint(self.complexity_range[0], self.complexity_range[1] + 1)
        
        # Generate equation with JAX
        jax_key = jax.random.PRNGKey(self.seed + idx)
        _, equation = self.arithmetic_process.generate(
            complexity, jax_key, self.max_prompt_length * 2, False
        )
        
        # Convert to numpy and create prompt
        equation_np = np.array(equation)
        
        # Find the first equals sign and truncate before it
        eql_positions = np.where(equation_np == self.eql_token)[0]
        if len(eql_positions) > 0:
            prompt_end = min(eql_positions[0], self.max_prompt_length)
        else:
            prompt_end = min(len(equation_np), self.max_prompt_length)
        
        prompt = equation_np[:prompt_end]
        
        # Pad to max length for batching
        if len(prompt) < self.max_prompt_length:
            padding = np.full(self.max_prompt_length - len(prompt), self.pad_token)
            prompt = np.concatenate([prompt, padding])
        
        # Convert to torch tensor
        prompt_tensor = torch.tensor(prompt, dtype=torch.long)
        
        # Create attention mask (1s for real tokens, 0s for padding)
        attention_mask = (prompt_tensor != self.pad_token).long()
        
        return {
            "input_ids": prompt_tensor,
            "attention_mask": attention_mask,
            "complexity": complexity,
        }


def create_rlvr_dataset(
    arithmetic_process: ArithmeticProcess,
    dataset_type: str = "prompt",
    **kwargs
) -> Dataset:
    """Factory function to create RLVR datasets.
    
    Args:
        arithmetic_process: The arithmetic process for generating data
        dataset_type: Type of dataset ("full" or "prompt")
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Dataset instance
    """
    if dataset_type == "full":
        return ArithmeticRLVRDataset(arithmetic_process, **kwargs)
    elif dataset_type == "prompt":
        return ArithmeticPromptDataset(arithmetic_process, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")