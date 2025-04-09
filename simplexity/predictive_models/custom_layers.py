from typing import Callable, Any
import dataclasses
from penzai import pz
import jax

@pz.pytree_dataclass
class WrapAndSummarize(pz.nn.Layer):
    """A layer that wraps another layer, caching a custom summarization of its output.

    Example:

    layer = WrapAndSummarize(wrapped_layer, summary_fn, identifier)
    filter_fn = WrapAndSummarize.get_select_fn(identifier)
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
    identifier: str = dataclasses.field(
        metadata={"pytree_node": False},
    )

    def __call__(self, argument: pz.nx.NamedArray, /, **side_inputs):
        """Performs the profile summary on the wrapped method output."""
        output = self.wrapped_layer(argument, **side_inputs)
        self.summary.value = self.summary_fn(output)
        return output

    def __post_init__(self):
        """this allows the get_state_vars to work"""
        self.summary.metadata["identifier"] = self.identifier

def get_state_vars(model: pz.nn.Layer, ident: str):
    """Retrieve the list of pz.StateVariables with metadata {"identifier": id}

    Use this function to retrieve the `summary` state variable for each
    WrapAndSummarize layer. 
    """
    def filter_fn(obj):
        return (isinstance(obj, pz.StateVariable) and 
                obj.metadata.get("identifier") == ident)
    return pz.select(model).at_subtrees_where(filter_fn).get_sequence()

