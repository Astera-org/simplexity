import dataclasses
from collections.abc import Callable
from typing import Any

from penzai import pz


@pz.pytree_dataclass
class ConcatInputs(pz.nn.Layer):
    """A Sequential layer that returns the stacked inputs of a subset of layers.

    target_layer_types: a tuple of layer types to collect inputs for.
    stack_axis: a name of the axis to stack across
    """

    model: pz.nn.Sequential
    target_layer_types: tuple[Any]
    stack_axis: str

    def __call__(self, argument: Any, /, **side_inputs) -> Any:
        """The 'forward' call method."""
        activations = []
        for layer in self.model.sublayers:
            if isinstance(layer, self.target_layer_types):
                if self.stack_axis not in argument.named_shape:
                    raise RuntimeError(
                        f"stack axis {self.stack_axis} not found in activation axes: {argument.named_shape.keys()}"
                    )
                activations.append(argument)
            argument = layer(argument, **side_inputs)
        return pz.nx.concatenate(activations, self.stack_axis)


@pz.pytree_dataclass
class SaveInput(pz.nn.Layer):
    """A pass-through layer which merely saves the value passed through."""

    saved: pz.StateVariable[Any] = dataclasses.field(default_factory=lambda: pz.StateVariable(None), init=False)
    tag: str = dataclasses.field(
        metadata={"pytree_node": False},
    )

    def __post_init__(self):
        """This allows the get_state_vars to work."""
        self.saved.metadata["tag"] = self.tag

    def __call__(self, argument: Any, /, **side_inputs) -> Any:
        """The 'forward' call method."""
        self.saved.value = argument
        return argument


@pz.pytree_dataclass
class SaveInputs(pz.nn.Layer):
    """Layer to save inputs."""

    saved: pz.StateVariable[list[Any]]

    def __call__(self, argument: Any, /, **side_inputs) -> Any:
        """Save inputs as a side effect."""
        self.saved.value = self.saved.value + [argument]
        return argument


@pz.pytree_dataclass
class WrapAndSummarize(pz.nn.Layer):
    """A layer that wraps another layer, caching a custom summarization of its output.

    Example:
    layer = WrapAndSummarize(wrapped_layer, summary_fn, tag)
    filter_fn = WrapAndSummarize.get_select_fn(tag)
    pz.select(...).where(filter_fn)
    """

    wrapped_layer: pz.nn.Layer
    summary_fn: Callable[[Any], Any] = dataclasses.field(
        metadata={"pytree_node": False}  # prevent jit tracing of this function
    )
    summary: pz.StateVariable[Any] = dataclasses.field(
        default_factory=lambda: pz.StateVariable(None),
        init=False,
    )
    tag: str = dataclasses.field(
        metadata={"pytree_node": False},
    )

    def __post_init__(self):
        """This allows the get_state_vars to work."""
        self.summary.metadata["tag"] = self.tag
        self.summary.label = f"{self.tag}_{id(self)}"

    def __call__(self, argument: Any, /, **side_inputs) -> Any:
        """Performs the profile summary on the wrapped method output."""
        output = self.wrapped_layer(argument, **side_inputs)
        self.summary.value = self.summary_fn(output)
        return output


def get_state_vars(model: pz.nn.Layer, *tags: list[str]) -> tuple[pz.StateVariableValue]:
    """Retrieves all StateVariableValue's with a metadata "tag" matching `tags`."""
    _, state_vars = pz.unbind_state_vars(model, lambda x: x.metadata.get("tag") in tags)
    return pz.freeze_state_vars(state_vars)
