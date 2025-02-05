import pytest
import torch
import numpy as np
from epsilon_transformers.analysis.activation_analysis import run_activation_to_beliefs_regression

def test_activation_to_beliefs_regression():
    # Create dummy data
    batch_size = 10
    n_ctx = 5
    d_model = 8
    belief_dim = 3
    
    activations = torch.randn(batch_size, n_ctx, d_model)
    ground_truth_beliefs = torch.randn(batch_size, n_ctx, belief_dim)
    
    # Run regression
    regression, predictions, mse, mse_shuffled, pred_shuffled, mse_cv, pred_cv, test_inds = \
        run_activation_to_beliefs_regression(activations, ground_truth_beliefs)
    
    # Test output shapes
    assert predictions.shape == ground_truth_beliefs.shape
    assert isinstance(mse, torch.Tensor)
    assert isinstance(mse_shuffled, torch.Tensor)
    assert isinstance(mse_cv, torch.Tensor)
    assert pred_shuffled.shape == ground_truth_beliefs.shape
    
    # Test that MSE is positive
    assert mse >= 0
    assert mse_shuffled >= 0
    assert mse_cv >= 0 