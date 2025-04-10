from enum import StrEnum


class ModelFramework(StrEnum):
    """The type of model."""

    Equinox = "equinox"
    Penzai = "penzai"
