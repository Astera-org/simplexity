"""
Cartesian Product Process - combines multiple HMMs/GHMMs efficiently.
Generates from each component independently and combines outputs only when needed.
"""

from typing import List, Union, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel


@dataclass
class ProductSpaceInfo:
    """Information about the product space encoding."""
    component_vocab_sizes: List[int]
    total_vocab_size: int
    component_bases: List[int]  # Multiplier for each component in encoding


class CartesianProductProcess:
    """
    A process that generates from multiple HMMs/GHMMs independently
    and combines their outputs into a product space.
    
    Key design: We keep all component processes separate and only
    combine outputs when needed. No giant transition matrices!
    """
    
    def __init__(
        self, 
        processes: List[Union[HiddenMarkovModel, GeneralizedHiddenMarkovModel]],
        names: Optional[List[str]] = None
    ):
        """
        Initialize with a list of component processes.
        
        Args:
            processes: List of HMM or GHMM instances
            names: Optional names for each process (for debugging/analysis)
        """
        if len(processes) == 0:
            raise ValueError("Need at least one process")
            
        self.processes = processes
        self.names = names or [f"Process_{i}" for i in range(len(processes))]
        
        # Calculate product space dimensions
        self.component_vocab_sizes = [p.vocab_size for p in processes]
        self.total_vocab_size = np.prod(self.component_vocab_sizes)
        
        # Calculate bases for encoding (each component's multiplier)
        # For example, if we have vocabs [3, 4, 2], then:
        # token = comp0 * (4*2) + comp1 * 2 + comp2
        self.component_bases = []
        for i in range(len(processes)):
            base = np.prod(self.component_vocab_sizes[i+1:]) if i < len(processes)-1 else 1
            self.component_bases.append(int(base))
        
        self.product_info = ProductSpaceInfo(
            component_vocab_sizes=self.component_vocab_sizes,
            total_vocab_size=int(self.total_vocab_size),
            component_bases=self.component_bases
        )
        
        # Combine initial states (just concatenate them)
        self.component_initial_states = [p.initial_state for p in processes]
        self.initial_state = jnp.concatenate(self.component_initial_states)
        
        # Store state boundaries for splitting concatenated states
        self.state_boundaries = []
        current = 0
        for p in processes:
            self.state_boundaries.append((current, current + p.num_states))
            current += p.num_states
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of the product space."""
        return self.product_info.total_vocab_size
    
    @property
    def num_states(self) -> int:
        """Total number of states (sum of all component states)."""
        return sum(p.num_states for p in self.processes)
    
    def encode_product_tokens(
        self, 
        component_sequences: List[jax.Array]
    ) -> jax.Array:
        """
        Encode sequences from each component into product space tokens.
        
        Args:
            component_sequences: List of arrays, each shape (batch_size, seq_len)
                                or (seq_len,) with tokens from each component
        
        Returns:
            Product space tokens with same shape as input sequences
        """
        # Start with zeros
        if component_sequences[0].ndim == 1:
            product_tokens = jnp.zeros_like(component_sequences[0])
        else:
            product_tokens = jnp.zeros_like(component_sequences[0])
        
        # Add each component's contribution
        for comp_seq, base in zip(component_sequences, self.component_bases):
            product_tokens = product_tokens + comp_seq * base
            
        return product_tokens.astype(jnp.int32)
    
    def decode_product_tokens(
        self,
        product_tokens: jax.Array
    ) -> List[jax.Array]:
        """
        Decode product space tokens back into component sequences.
        
        Args:
            product_tokens: Array of product space tokens
            
        Returns:
            List of component sequences
        """
        component_sequences = []
        remaining = product_tokens
        
        for i, (vocab_size, base) in enumerate(zip(
            self.component_vocab_sizes, 
            self.component_bases
        )):
            # Extract this component's tokens
            comp_tokens = remaining // base
            component_sequences.append(comp_tokens)
            
            # Remove this component's contribution
            remaining = remaining % base
            
        return component_sequences
    
    def split_states(self, combined_states: jax.Array) -> List[jax.Array]:
        """
        Split concatenated states back into component states.
        
        Args:
            combined_states: Shape (batch_size, total_num_states) or (total_num_states,)
            
        Returns:
            List of component states
        """
        component_states = []
        for start, end in self.state_boundaries:
            if combined_states.ndim == 1:
                component_states.append(combined_states[start:end])
            else:
                component_states.append(combined_states[:, start:end])
        return component_states
    
    def combine_states(self, component_states: List[jax.Array]) -> jax.Array:
        """
        Combine component states into a single concatenated state.
        
        Args:
            component_states: List of states from each component
            
        Returns:
            Concatenated state array
        """
        if component_states[0].ndim == 1:
            return jnp.concatenate(component_states)
        else:
            return jnp.concatenate(component_states, axis=1)
    
    def generate_components(
        self,
        batch_size: int,
        seq_len: int,
        key: jax.random.PRNGKey,
        bos_token: Optional[int] = None,
        eos_token: Optional[int] = None
    ) -> Tuple[List[jax.Array], List[jax.Array], List[jax.Array]]:
        """
        Generate sequences from each component independently.
        
        Returns:
            (component_states, component_inputs, component_labels)
            Each is a list with one array per component process
        """
        from simplexity.generative_processes.generator import generate_data_batch
        
        # Split the key for each process
        keys = jax.random.split(key, len(self.processes))
        
        component_states = []
        component_inputs = []
        component_labels = []
        
        for process, subkey in zip(self.processes, keys):
            # Initialize states for this component
            init_state = process.initial_state
            states = jnp.repeat(init_state[None, :], batch_size, axis=0)
            
            # Generate data for this component
            new_states, inputs, labels = generate_data_batch(
                states, process, batch_size, seq_len, subkey,
                bos_token=bos_token, eos_token=eos_token
            )
            
            component_states.append(new_states)
            component_inputs.append(inputs)
            component_labels.append(labels)
            
        return component_states, component_inputs, component_labels
    
    def generate_product_batch(
        self,
        batch_size: int,
        seq_len: int,
        key: jax.random.PRNGKey,
        bos_token: Optional[int] = None,
        eos_token: Optional[int] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Generate a batch in the product space.
        
        Returns:
            (combined_states, product_inputs, product_labels)
            Where product tokens are encoded in the product space
        """
        # Generate from each component
        comp_states, comp_inputs, comp_labels = self.generate_components(
            batch_size, seq_len, key, bos_token, eos_token
        )
        
        # Combine states
        combined_states = self.combine_states(comp_states)
        
        # Encode into product space
        product_inputs = self.encode_product_tokens(comp_inputs)
        product_labels = self.encode_product_tokens(comp_labels)
        
        return combined_states, product_inputs, product_labels
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        components_str = ", ".join([
            f"{name}(vocab={vs})" 
            for name, vs in zip(self.names, self.component_vocab_sizes)
        ])
        return (
            f"CartesianProductProcess("
            f"components=[{components_str}], "
            f"total_vocab={self.vocab_size})"
        )