import dataclasses
from collections.abc import Callable
from typing import Any

from penzai import pz


@pz.pytree_dataclass
class ConcatInputs(pz.nn.Layer):
    """A Sequential layer that returns the stacked inputs of a subset of layers...

    target_layer_types: a tuple of layer types to collect inputs for.
    stack_axis: a name of the axis to stack across
    """

    model: pz.nn.Sequential
    target_layer_types: tuple[Any]
    stack_axis: str

    def __call__(self, x, /, **side_inputs):
        """The 'forward' call method."""
        activations = []
        for layer in self.model.sublayers:
            if isinstance(layer, self.target_layer_types):
                if self.stack_axis not in x.named_shape:
                    raise RuntimeError(
                        f"stack axis {self.stack_axis} not found in activation axes: {x.named_shape.keys()}"
                    )
                activations.append(x)
            x = layer(x, **side_inputs)
        out = pz.nx.concatenate(activations, self.stack_axis)
        return out


@pz.pytree_dataclass
class SaveInput(pz.nn.Layer):
    """A pass-through layer which merely saves the value passed through."""

    saved: pz.StateVariable[Any] = dataclasses.field(default_factory=lambda: pz.StateVariable(None), init=False)
    tag: str = dataclasses.field(
        metadata={"pytree_node": False},
    )

    def __call__(self, x: pz.nx.NamedArray, /, **side_inputs):
        """The 'forward' call method."""
        self.saved.value = x
        return x

    def __post_init__(self):
        """This allows the get_state_vars to work."""
        self.saved.metadata["tag"] = self.tag


@pz.pytree_dataclass
class WrapAndSummarize(pz.nn.Layer):
    """A layer that wraps another layer, caching a custom summarization of its output.

    Example:
    layer = WrapAndSummarize(wrapped_layer, summary_fn, tag)
    filter_fn = WrapAndSummarize.get_select_fn(tag)
    pz.select(...).where(filter_fn)
    """

    wrapped_layer: pz.nn.Layer
    summary_fn: Callable[[pz.nx.NamedArray], pz.nx.NamedArray] = dataclasses.field(
        # prevent jit tracing of this function
        metadata={"pytree_node": False}
    )
    summary: pz.StateVariable[Any] = dataclasses.field(
        default_factory=lambda: pz.StateVariable(None),
        init=False,
    )
    # Provide this to retrieve
    tag: str = dataclasses.field(
        metadata={"pytree_node": False},
    )

    def __call__(self, argument: pz.nx.NamedArray, /, **side_inputs):
        """Performs the profile summary on the wrapped method output."""
        output = self.wrapped_layer(argument, **side_inputs)
        self.summary.value = self.summary_fn(output)
        return output

    def __post_init__(self):
        """This allows the get_state_vars to work."""
        self.summary.metadata["tag"] = self.tag
        self.summary.label = f"{self.tag}_{id(self)}"


def get_state_vars(model: pz.nn.Layer, *tags: list[str]) -> tuple[pz.StateVariableValue]:
    """Retrieves all StateVariableValue's with a metadata "tag" matching `tags`."""
    _, state_vars = pz.unbind_state_vars(model, lambda x: x.metadata.get("tag") in tags)
    return pz.freeze_state_vars(state_vars)
