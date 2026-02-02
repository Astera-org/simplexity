"""Tests for TreeConditional structure."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    build_factored_process_from_spec,
    build_tree_from_spec,
)
from simplexity.generative_processes.structures import SequentialConditional, TreeConditional


class TestTreeConditionalBasics:
    """Test basic TreeConditional functionality."""

    def test_fanout_structure(self):
        """Test fan-out structure: Factor 0 is parent of both Factor 1 and Factor 2."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],  # 3 tokens -> 2 variants
            },
            {
                "component_type": "hmm",
                "depends_on": 0,  # Also depends on Factor 0, not Factor 1
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [1, 0, 1],
            },
        ]

        process = build_factored_process_from_spec(structure_type="tree", spec=spec)

        assert isinstance(process.structure, TreeConditional)
        assert process.structure.parent_indices == (None, 0, 0)
        assert len(process.structure.control_maps) == 3
        assert process.structure.control_maps[0] is None
        chex.assert_trees_all_equal(process.structure.control_maps[1], jnp.array([0, 1, 0]))
        chex.assert_trees_all_equal(process.structure.control_maps[2], jnp.array([1, 0, 1]))

    def test_chain_as_tree(self):
        """Test that chain structure can be expressed as tree: 0 -> 1 -> 2."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
            {
                "component_type": "hmm",
                "depends_on": 1,  # Depends on Factor 1, forming chain
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
        ]

        process = build_factored_process_from_spec(structure_type="tree", spec=spec)

        assert process.structure.parent_indices == (None, 0, 1)


class TestTreeConditionalSelectVariants:
    """Test variant selection in TreeConditional."""

    def test_select_variants_fanout(self):
        """Test variant selection for fan-out structure."""
        parent_indices = (None, 0, 0)
        control_maps = (
            None,
            jnp.array([0, 1, 0]),  # Factor 1: parent token -> variant
            jnp.array([1, 0, 1]),  # Factor 2: parent token -> variant
        )
        vocab_sizes = jnp.array([3, 3, 3])

        tree = TreeConditional(parent_indices, control_maps, vocab_sizes)

        # Create mock context (unused by select_variants)
        from simplexity.generative_processes.structures.protocol import ConditionalContext

        mock_context = ConditionalContext(
            states=tuple(),
            component_types=tuple(),
            transition_matrices=tuple(),
            normalizing_eigenvectors=tuple(),
            vocab_sizes=jnp.array([]),
            num_variants=tuple(),
        )

        # Test with parent token 0
        obs = (jnp.array(0), jnp.array(1), jnp.array(2))
        variants = tree.select_variants(obs, mock_context)
        assert variants[0] == 0  # Root always variant 0
        assert variants[1] == 0  # control_maps[1][0] = 0
        assert variants[2] == 1  # control_maps[2][0] = 1

        # Test with parent token 1
        obs = (jnp.array(1), jnp.array(0), jnp.array(0))
        variants = tree.select_variants(obs, mock_context)
        assert variants[0] == 0  # Root always variant 0
        assert variants[1] == 1  # control_maps[1][1] = 1
        assert variants[2] == 0  # control_maps[2][1] = 0

        # Test with parent token 2
        obs = (jnp.array(2), jnp.array(0), jnp.array(0))
        variants = tree.select_variants(obs, mock_context)
        assert variants[0] == 0  # Root always variant 0
        assert variants[1] == 0  # control_maps[1][2] = 0
        assert variants[2] == 1  # control_maps[2][2] = 1

    def test_select_variants_chain(self):
        """Test variant selection for chain structure expressed as tree."""
        parent_indices = (None, 0, 1)
        control_maps = (
            None,
            jnp.array([0, 1, 0]),  # Factor 1 depends on Factor 0
            jnp.array([1, 0, 1]),  # Factor 2 depends on Factor 1
        )
        vocab_sizes = jnp.array([3, 3, 3])

        tree = TreeConditional(parent_indices, control_maps, vocab_sizes)

        from simplexity.generative_processes.structures.protocol import ConditionalContext

        mock_context = ConditionalContext(
            states=tuple(),
            component_types=tuple(),
            transition_matrices=tuple(),
            normalizing_eigenvectors=tuple(),
            vocab_sizes=jnp.array([]),
            num_variants=tuple(),
        )

        # Test: parent token 0 for factor 0, token 1 for factor 1
        obs = (jnp.array(0), jnp.array(1), jnp.array(2))
        variants = tree.select_variants(obs, mock_context)
        assert variants[0] == 0  # Root
        assert variants[1] == 0  # control_maps[1][0] = 0
        assert variants[2] == 0  # control_maps[2][1] = 0 (based on factor 1's token)


class TestTreeConditionalJointDistribution:
    """Test joint distribution computation in TreeConditional."""

    def test_joint_distribution_sums_to_one(self):
        """Test that joint distribution sums to 1."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [1, 0, 1],
            },
        ]

        process = build_factored_process_from_spec(structure_type="tree", spec=spec)
        joint = process.observation_probability_distribution(process.initial_states)

        chex.assert_trees_all_close(joint.sum(), 1.0, atol=1e-6)

    def test_chain_tree_matches_sequential(self):
        """Test that chain expressed as tree matches SequentialConditional output."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
        ]

        # Build chain version
        chain_process = build_factored_process_from_spec(structure_type="chain", spec=spec)

        # Build tree version with depends_on
        tree_spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
        ]
        tree_process = build_factored_process_from_spec(structure_type="tree", spec=tree_spec)

        # Compare joint distributions
        chain_joint = chain_process.observation_probability_distribution(chain_process.initial_states)
        tree_joint = tree_process.observation_probability_distribution(tree_process.initial_states)

        chex.assert_trees_all_close(chain_joint, tree_joint, atol=1e-6)


class TestBuildTreeFromSpec:
    """Test build_tree_from_spec function."""

    def test_invalid_depends_on_negative(self):
        """Test that negative depends_on raises error."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": -1,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                ],
                "control_map": [0, 1, 0],
            },
        ]

        with pytest.raises(ValueError, match="must be in range"):
            build_tree_from_spec(spec)

    def test_invalid_depends_on_forward_reference(self):
        """Test that forward reference in depends_on raises error."""
        spec = [
            {
                "component_type": "hmm",
                "depends_on": 1,  # Invalid: references factor 1 which comes later
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
                "control_map": [0, 1, 0],
            },
            {
                "component_type": "hmm",
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                ],
            },
        ]

        with pytest.raises(ValueError, match="must be in range"):
            build_tree_from_spec(spec)

    def test_missing_control_map_with_depends_on(self):
        """Test that missing control_map when depends_on is specified raises error."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                ],
                # Missing control_map
            },
        ]

        with pytest.raises(ValueError, match="control_map is required"):
            build_tree_from_spec(spec)

    def test_control_map_length_mismatch(self):
        """Test that control_map length must match parent vocab size."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                ],
                "control_map": [0, 1],  # Wrong length - mess3 has vocab 3
            },
        ]

        with pytest.raises(ValueError, match="must equal parent"):
            build_tree_from_spec(spec)

    def test_empty_spec_raises_error(self):
        """Test that empty spec raises error."""
        with pytest.raises(ValueError, match="must contain at least one node"):
            build_tree_from_spec([])


class TestTreeWithHiddenFactors:
    """Test TreeConditional with hidden factors."""

    def test_hidden_root_fanout(self):
        """Test fan-out with hidden root factor."""
        spec = [
            {
                "component_type": "hmm",
                "hidden": True,  # Hidden root
                "variants": [{"process_name": "leaky_rrxor", "process_params": {"p1": 0.5, "p2": 0.5, "epsilon": 0.0}}],
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [0, 1],  # rrxor has vocab 2
            },
            {
                "component_type": "hmm",
                "depends_on": 0,
                "variants": [
                    {"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}},
                    {"process_name": "mess3", "process_params": {"x": 0.5, "a": 0.6}},
                ],
                "control_map": [1, 0],
            },
        ]

        process = build_factored_process_from_spec(structure_type="tree", spec=spec)

        assert process.hidden_factor_indices == frozenset({0})
        assert isinstance(process.structure, TreeConditional)

        # Joint should sum to 1
        joint = process.observation_probability_distribution(process.initial_states)
        chex.assert_trees_all_close(joint.sum(), 1.0, atol=1e-6)

        # Observable joint should have shape [V1 * V2] = [3 * 3] = 9
        # (hidden factor marginalized out)
        assert joint.shape == (9,)
