import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNWrapper(nn.Module):
    """Wrapper class for RNN to match Transformer interface"""
    
    def __init__(self, rnn: nn.RNN, output_layer: nn.Linear):
        super().__init__()
        self.rnn = rnn
        self.output_layer = output_layer
        self.vocab_size = output_layer.out_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching Transformer interface
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Logits tensor of shape (batch_size, seq_length, vocab_size)
        """
        # Convert input tokens to one-hot vectors
        one_hot = F.one_hot(x.to(torch.int64), num_classes=self.vocab_size).float()
        
        # Run RNN
        output, _ = self.rnn(one_hot)
        
        # Project to vocabulary size
        return self.output_layer(output)
    
    def forward_with_hidden(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns hidden states
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        one_hot = F.one_hot(x.to(torch.int64), num_classes=self.vocab_size).float()
        output, hidden = self.rnn(one_hot, hidden)
        return self.output_layer(output), hidden