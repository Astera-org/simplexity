"""Visualization subpackage for activation analysis."""

from simplexity.activations.visualization.data_structures import (
    _SCALAR_INDEX_SENTINEL,
    ActivationVisualizationPayload,
    PreparedMetadata,
    VisualizationControlDetail,
    VisualizationControlsState,
)
from simplexity.activations.visualization.dataframe_builders import (
    _build_dataframe,
    _build_metadata_columns,
)
from simplexity.activations.visualization.field_resolution import (
    _lookup_projection_array,
    _lookup_scalar_value,
    _maybe_component,
    _resolve_belief_states,
    _resolve_field,
)
from simplexity.activations.visualization.pattern_expansion import (
    _expand_field_mapping,
    _has_field_pattern,
    _has_key_pattern,
    _parse_component_spec,
)
from simplexity.activations.visualization.preprocessing import (
    _apply_preprocessing,
)

__all__ = [
    "ActivationVisualizationPayload",
    "PreparedMetadata",
    "VisualizationControlDetail",
    "VisualizationControlsState",
    "_SCALAR_INDEX_SENTINEL",
    "_apply_preprocessing",
    "_build_dataframe",
    "_build_metadata_columns",
    "_expand_field_mapping",
    "_has_field_pattern",
    "_has_key_pattern",
    "_lookup_projection_array",
    "_lookup_scalar_value",
    "_maybe_component",
    "_parse_component_spec",
    "_resolve_belief_states",
    "_resolve_field",
]
