from typing import Any

from penzai import pz


@pz.pytree_dataclass
class SaveActivations(pz.nn.Layer):
    """Layer to save activations."""

    saved_activations: pz.StateVariable[list[Any]]

    def __call__(self, activations: Any, **unused_side_inputs) -> Any:
        """Save activations as a side effect."""
        self.saved_activations.value = self.saved_activations.value + [activations]
        return activations
