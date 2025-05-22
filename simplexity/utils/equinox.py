from collections.abc import Callable
from typing import Any

import equinox as eqx


def vmap_model(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a function to use a penzai model."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "model" not in kwargs:
            raise ValueError("Function must be called with 'model' keyword argument")
        vmapped_model = eqx.filter_vmap(kwargs["model"])
        kwargs["model"] = vmapped_model
        return f(*args, **kwargs)

    return wrapper
