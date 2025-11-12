"""Types for predictive models."""

from enum import StrEnum
from functools import singledispatch
from typing import Any

import equinox as eqx


class ModelFramework(StrEnum):
    """The type of model."""

    EQUINOX = "equinox"
    PENZAI = "penzai"
    PYTORCH = "pytorch"


@singledispatch
def get_model_framework(predictive_model: Any) -> ModelFramework:
    """Get the model framework of a predictive model."""
    raise ValueError(f"Unsupported model framework: {type(predictive_model)}")


@get_model_framework.register(eqx.Module)
def _(predictive_model: eqx.Module) -> ModelFramework:
    return ModelFramework.EQUINOX


try:
    from torch.nn import Module as TorchModule
except ImportError:  # torch is optional
    pass
else:

    @get_model_framework.register(TorchModule)
    def _(predictive_model: TorchModule) -> ModelFramework:
        return ModelFramework.PYTORCH


try:
    from penzai.nn.layer import Layer as PenzaiModel
except ImportError:  # penzai is optional
    pass
else:

    @get_model_framework.register(PenzaiModel)
    def _(predictive_model: PenzaiModel) -> ModelFramework:
        return ModelFramework.PENZAI
