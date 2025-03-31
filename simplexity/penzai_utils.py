from dataclasses import dataclass

from penzai import pz
from penzai.core.named_axes import NamedArray
from penzai.core.struct import Struct


@dataclass
class ParamCountNode:
    """A node in the parameter count tree."""

    name: str
    param_count: int
    children: list["ParamCountNode"]


def get_parameter_count_tree(struct: Struct) -> ParamCountNode:
    """Get a tree of the parameter counts for a struct."""
    _, params = pz.unbind_params(struct)
    root = ParamCountNode(name="", param_count=0, children=[])
    for param in params:
        named_array: NamedArray = param.value
        param_count = named_array.data_array.size
        label = str(param.label)
        label_parts = label.split("/")
        current = root
        for part in label_parts:
            children_parts = [i.name for i in current.children]
            try:
                i = children_parts.index(part)
            except ValueError:
                current.children.append(ParamCountNode(name=part, param_count=0, children=[]))
                i = len(current.children) - 1
            current.param_count += param_count
            current = current.children[i]
        current.param_count += param_count
    if len(root.children) == 1:
        return root.children[0]
    return root
