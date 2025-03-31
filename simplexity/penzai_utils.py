import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

from penzai.core.named_axes import NamedArray
from penzai.core.struct import Struct
from penzai.core.variables import Parameter


def get_parameter_count(x: Struct | list[Any] | Parameter) -> int:
    """Recursively count the number of parameters in a Penzai structure."""
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
    """Recursively count the number of parameters in a Penzai structure."""
    if isinstance(x, Struct):
        subtrees = x.tree_flatten()[0]
        if not isinstance(subtrees, Struct | list | Parameter):
            return ParameterTree()
        subtree = get_parameter_tree(subtrees)
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
        subtrees = [get_parameter_tree(i) for i in x if isinstance(i, Struct | list | Parameter)]
        subtrees = [subtree for subtree in subtrees if subtree != ParameterTree()]
        parameters = sum(subtree.parameters for subtree in subtrees)
        if parameters == 0:
            return ParameterTree()
        name = x.__class__.__name__
        if len(subtrees) == 1:
            child_name = subtrees[0].name
            if name == list.__name__:
                name = child_name
            elif child_name not in (list.__name__, Parameter.__name__):
                name = f"{name}.{child_name}"
            subtrees = subtrees[0].children
        return ParameterTree(
            name=name,
            parameters=parameters,
            children=subtrees,
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
