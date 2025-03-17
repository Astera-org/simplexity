# MSP Golden Tests

This folder contains golden test files for Mixed State Presentations (MSPs) of various stochastic processes. These files are used to validate new implementations against known-good reference outputs.

## File Structure

Each `.npz` file represents the MSP for a specific process with specific parameters. The naming convention is:

```
<process_name>_<param1>_<value1>_<param2>_<value2>.npz
```

For example:
- `fanizza_alpha_2000_lamb_0.49.npz`
- `mess3_x_0.15_a_0.6.npz`
- `rrxor_pR1_0.5_pR2_0.5.npz`

## File Contents

Each `.npz` file contains the following arrays:

- `paths`: A 2D array where each row is a path through the process, padded with -1 values
- `probs`: Array of probabilities for each path
- `beliefs`: A 2D array where each row is a belief state vector corresponding to a path

## Loading and Using the Files

```python
import numpy as np

# Load the file
data = np.load('fanizza_alpha_2000_lamb_0.49.npz')

# Extract components
paths = data['paths']
probs = data['probs']
beliefs = data['beliefs']
```

## Utility Functions

```python
# Find the belief state for a specific path
def find_belief_for_path(paths, beliefs, path_to_find):
    # Convert path to list if it's a tuple
    if isinstance(path_to_find, tuple):
        path_to_find = list(path_to_find)
        
    # Find matching row in paths array
    for i, path in enumerate(paths):
        valid_path = [p for p in path if p >= 0]  # Remove padding
        if valid_path == path_to_find:
            return beliefs[i]
    
    return None

# Reconstruct the original node_info dictionary
def reconstruct_node_info(npz_file):
    data = np.load(npz_file)
    paths = data['paths']
    probs = data['probs']
    beliefs = data['beliefs']
    
    node_info = {}
    for i in range(len(paths)):
        # Convert path array to tuple, ignoring padding (-1 values)
        path = tuple(int(x) for x in paths[i] if x >= 0)
        
        node_info[path] = {
            'path_prob': float(probs[i]),
            'belief_state': beliefs[i].reshape(1, -1)  # Reshape to match original format
        }
    
    return node_info
```

## Testing Your Implementation

To use these files as golden tests:

1. Generate the same process in your implementation using the same parameters
2. Create a comparable data structure (MSP or equivalent)
3. Compare the paths, belief states, and path probabilities with those in the golden file
4. Ensure numerical values match within an acceptable tolerance (e.g., 1e-5)

Example test:

```python
def test_against_golden(your_msp, golden_file, tolerance=1e-5):
    # Load golden data
    golden_data = np.load(golden_file)
    golden_paths = golden_data['paths']
    golden_probs = golden_data['probs']
    golden_beliefs = golden_data['beliefs']
    
    # Get your paths and beliefs in comparable format
    your_paths = [p for p in your_msp.get_all_paths()]
    your_probs = [your_msp.get_path_probability(p) for p in your_paths]
    your_beliefs = [your_msp.get_belief_state(p).flatten() for p in your_paths]
    
    # Check number of states matches
    assert len(your_paths) == len(golden_paths), "Number of paths doesn't match"
    
    # Match paths and compare beliefs and probabilities
    for i, golden_path in enumerate(golden_paths):
        cleaned_path = tuple(int(x) for x in golden_path if x >= 0)
        
        # Find matching path in your implementation
        matching_idx = next((j for j, p in enumerate(your_paths) if p == cleaned_path), None)
        assert matching_idx is not None, f"Path {cleaned_path} not found in implementation"
        
        # Compare belief state
        assert np.allclose(your_beliefs[matching_idx], golden_beliefs[i], atol=tolerance), \
            f"Belief states don't match for path {cleaned_path}"
            
        # Compare path probability
        assert np.isclose(your_probs[matching_idx], golden_probs[i], atol=tolerance), \
            f"Path probabilities don't match for path {cleaned_path}"
    
    print("Test passed! Implementation matches golden file.")
```

## Source Information

These golden files were generated from the epsilon-transformers repository using the original implementation of MSP. The MSP trees were derived from `TransitionMatrixGHMM` objects with a depth of 4.
