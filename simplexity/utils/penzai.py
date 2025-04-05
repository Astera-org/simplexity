from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
from penzai import pz
from penzai.core.named_axes import AxisName, NamedArray
from penzai.core.struct import Struct
from penzai.core.variables import (
    AbstractVariableValue,
    LabeledVariableValue,
    ParameterValue,
    StateVariableValue,
    VariableLabel,
)


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
        for name in label_parts:
            children_names = [i.name for i in current.children]
            try:
                i = children_names.index(name)
            except ValueError:
                current.children.append(ParamCountNode(name=name, param_count=0, children=[]))
                i = len(current.children) - 1
            current.param_count += param_count
            current = current.children[i]
        current.param_count += param_count
    if len(root.children) == 1:
        return root.children[0]
    return root


class VariableValueClass(StrEnum):
    """The class of a penzai variable value.

    https://penzai.readthedocs.io/en/latest/_autosummary/leaf/penzai.core.variables.AbstractVariableValue.html
    """

    PARAMETER = "parameter"
    STATE_VARIABLE = "state_variable"


def deconstruct_variables(variable_values: tuple[AbstractVariableValue, ...]) -> Mapping[str, Any]:
    """Decompose a tree into a mapping of items orbax can save."""
    variable_value_classes: list[VariableValueClass] = []
    variable_labels: list[VariableLabel] = []
    axis_names: list[tuple[AxisName, ...]] = []
    axis_sizes: list[tuple[int, ...]] = []
    metadata: list[dict[Any, Any]] = []

    for variable_value in variable_values:
        assert isinstance(variable_value, LabeledVariableValue)
        variable_labels.append(variable_value.label)
        if isinstance(variable_value.value, NamedArray):
            axis_names.append(tuple(variable_value.value.named_axes.keys()))
            axis_sizes.append(tuple(variable_value.value.named_axes.values()))
        else:
            axis_names.append(())
            axis_sizes.append(())
        if isinstance(variable_value, ParameterValue):
            variable_value_classes.append(VariableValueClass.PARAMETER)
        elif isinstance(variable_value, StateVariableValue):
            variable_value_classes.append(VariableValueClass.STATE_VARIABLE)
        else:
            raise ValueError(f"Unknown variable type: {type(variable_value)}")
        metadata.append(variable_value.metadata)

    return {
        "data_arrays": tuple(variable_values),
        "axis_names": tuple(axis_names),
        "axis_sizes": tuple(axis_sizes),
        "variable_value_classes": tuple(variable_value_classes),
        "variable_labels": tuple(variable_labels),
        "metadata": tuple(metadata),
    }


def reconstruct_variables(items: Mapping[str, Any]) -> tuple[LabeledVariableValue, ...]:
    """Reconstruct variables from a mapping of items orbax can save."""
    data_arrays: tuple[jax.Array, ...] = items["data_arrays"]
    variable_value_classes: tuple[VariableValueClass, ...] = items["variable_value_classes"]
    variable_labels: tuple[VariableLabel, ...] = items["variable_labels"]
    axis_names: tuple[tuple[AxisName, ...], ...] = items["axis_names"]
    axis_sizes: tuple[tuple[int, ...], ...] = items["axis_sizes"]
    metadata: tuple[dict[Any, Any], ...] = items["metadata"]

    loaded_variables: list[LabeledVariableValue] = []
    for data_array, variable_value_class, variable_label, axis_names_, axis_sizes_, metadata_ in zip(
        data_arrays,
        variable_value_classes,
        variable_labels,
        axis_names,
        axis_sizes,
        metadata,
        strict=True,
    ):
        if axis_names_:
            named_axes = OrderedDict(zip(axis_names_, axis_sizes_, strict=True))
            value = NamedArray(named_axes, data_array)
        else:
            value = data_array
        if variable_value_class == VariableValueClass.PARAMETER:
            variable = ParameterValue(variable_label, value, metadata_)
        elif variable_value_class == VariableValueClass.STATE_VARIABLE:
            variable = StateVariableValue(variable_label, value, metadata_)
        else:
            raise ValueError(f"Unknown variable value class: {variable_value_class}")
        loaded_variables.append(variable)

    return tuple(loaded_variables)
