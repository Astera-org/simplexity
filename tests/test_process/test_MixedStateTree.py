import pytest
import numpy as np
from epsilon_transformers.process.MixedStateTree import MixedStateTree, MixedStateTreeNode

def test_mixed_state_tree_node():
    # Test node creation and child addition
    state_vector = np.array([0.6, 0.4])
    node = MixedStateTreeNode(
        state_prob_vector=state_vector,
        unnorm_state_prob_vector=state_vector,
        children=set(),
        path=[0],
        emission_prob=0.5,
        path_prob=1.0
    )
    
    assert np.allclose(node.state_prob_vector, state_vector)
    assert len(node.children) == 0
    assert node.path == [0]
    assert node.emission_prob == 0.5
    
    # Test adding child
    child_node = MixedStateTreeNode(
        state_prob_vector=np.array([0.3, 0.7]),
        unnorm_state_prob_vector=np.array([0.3, 0.7]),
        children=set(),
        path=[0, 1],
        emission_prob=0.3,
        path_prob=0.5
    )
    node.add_child(child_node)
    assert len(node.children) == 1

def test_mixed_state_tree():
    # Create a simple tree with root and one child
    root = MixedStateTreeNode(
        state_prob_vector=np.array([0.5, 0.5]),
        unnorm_state_prob_vector=np.array([0.5, 0.5]),
        children=set(),
        path=[],
        emission_prob=0.0,
        path_prob=1.0
    )
    
    child = MixedStateTreeNode(
        state_prob_vector=np.array([0.7, 0.3]),
        unnorm_state_prob_vector=np.array([0.7, 0.3]),
        children=set(),
        path=[0],
        emission_prob=0.5,
        path_prob=0.5
    )
    
    root.add_child(child)
    tree = MixedStateTree(root_node=root, process="test", nodes={root, child}, depth=1)
    
    assert len(tree.nodes) == 2
    assert tree.depth == 1
    assert len(tree.paths) == 2 