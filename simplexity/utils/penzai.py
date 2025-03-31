import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

from penzai.core.named_axes import NamedArray
from penzai.core.struct import Struct
from penzai.core.variables import Parameter
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig


def calculate_llamalike_transformer_parameter_count(config: LlamalikeTransformerConfig) -> int:
    """Calculate the number of parameters in a Penzai transformer."""
    return (
        1  # final layer norm
        + 2 * config.vocab_size  # embedding table and LM head weights
        + (  # decoder block
            2  # layer norms before attention and MLP
            + 2 * config.num_kv_heads * (1 + config.query_head_multiplier) * config.projection_dim  # QKV+output weights
            + 3 * config.mlp_hidden_dim  # gating, value, and output weights
        )
        * config.num_decoder_blocks
    ) * config.embedding_dim


def get_parameter_count(x: Struct | list[Any] | Parameter) -> int:
    """Count the total number of parameters in a Penzai structure."""
    if isinstance(x, Struct):
        children = x.tree_flatten()[0]
        if not isinstance(children, Struct | list | Parameter):
            return 0
        return get_parameter_count(children)

    if isinstance(x, list):
        return sum(get_parameter_count(i) for i in x if isinstance(i, Struct | list | Parameter))

    assert isinstance(x, Parameter)
    named_array: NamedArray = x.value
    return named_array.data_array.size


@dataclass
class ParameterTree:
    """A tree of parameters."""

    name: str = ""
    parameters: int = 0
    children: list["ParameterTree"] = field(default_factory=list)


def get_parameter_tree(x: Struct | list[Any] | Parameter) -> ParameterTree:
    """Construct a parameter tree from a Penzai structure."""
    if isinstance(x, Struct):
        children = x.tree_flatten()[0]
        if not isinstance(children, Struct | list | Parameter):
            return ParameterTree()
        subtree = get_parameter_tree(children)
        if subtree == ParameterTree():
            return ParameterTree()
        name = x.__class__.__name__
        if subtree.name not in (list.__name__, Parameter.__name__):
            name = f"{name}.{subtree.name}"
        return ParameterTree(
            name=name,
            parameters=subtree.parameters,
            children=subtree.children,
        )

    if isinstance(x, list):
        children = [get_parameter_tree(i) for i in x if isinstance(i, Struct | list | Parameter)]
        children = [child for child in children if child != ParameterTree()]
        parameters = sum(child.parameters for child in children)
        if parameters == 0:
            return ParameterTree()
        name = x.__class__.__name__
        if len(children) == 1:
            child_name = children[0].name
            if name == list.__name__:
                name = child_name
            elif child_name not in (list.__name__, Parameter.__name__):
                name = f"{name}.{child_name}"
            children = children[0].children
        return ParameterTree(
            name=name,
            parameters=parameters,
            children=children,
        )

    assert isinstance(x, Parameter)
    named_array: NamedArray = x.value
    parameters = named_array.data_array.size
    return ParameterTree(
        name=Parameter.__name__,
        parameters=parameters,
        children=[],
    )


@dataclass
class NamedParameters:
    """A named parameter."""

    name: str
    parameters: int


def get_parameter_list(parameter_tree: ParameterTree) -> list[NamedParameters]:
    """Get a list of named parameters from a parameter tree."""
    if not parameter_tree.children:
        return [NamedParameters(name=parameter_tree.name, parameters=parameter_tree.parameters)]

    parameter_list = list(itertools.chain(*map(get_parameter_list, parameter_tree.children)))

    def modify_name(named_param: NamedParameters) -> NamedParameters:
        return dataclasses.replace(named_param, name=f"{parameter_tree.name}.{named_param.name}")

    return list(map(modify_name, parameter_list))
