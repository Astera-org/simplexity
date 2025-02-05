import pytest
import torch
import torch.nn as nn
from simplexity.training.networks import RNNWrapper, create_RNN

def test_rnn_wrapper():
    vocab_size = 10
    hidden_size = 16
    num_layers = 2
    batch_size = 4
    seq_len = 8
    
    # Create RNN and wrapper
    rnn = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
    output_layer = nn.Linear(hidden_size, vocab_size)
    wrapper = RNNWrapper(rnn, output_layer)
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = wrapper(x)
    
    assert output.shape == (batch_size, seq_len, vocab_size)
    
def test_create_rnn():
    config = {
        'model_config': {
            'rnn_type': 'LSTM',
            'hidden_size': 16,
            'num_layers': 2,
            'dropout': 0.1
        }
    }
    vocab_size = 10
    device = 'cpu'
    
    model = create_RNN(config, vocab_size, device)
    
    assert isinstance(model, RNNWrapper)
    assert model.vocab_size == vocab_size
    assert model.rnn.hidden_size == 16
    assert model.rnn.num_layers == 2 