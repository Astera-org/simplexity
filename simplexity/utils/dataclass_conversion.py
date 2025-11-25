"""Utilities for converting OmegaConf sections into dataclass instances."""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def convert_to_dataclass(cfg_section: DictConfig | dict[str, Any] | Any, schema: type[T]) -> T:
    """Convert a nested DictConfig/dict payload into a dataclass instance."""

    if isinstance(cfg_section, schema):
        return cfg_section

    if isinstance(cfg_section, DictConfig):
        data = OmegaConf.to_container(cfg_section, resolve=True) or {}
    else:
        data = cfg_section

    if not isinstance(data, dict):
        return data  # type: ignore[return-value]

    return _dict_to_dataclass(data, schema)


def _dict_to_dataclass(data: dict[str, Any], schema: type[T]) -> T:
    if not is_dataclass(schema):
        return data  # type: ignore[return-value]

    try:
        type_hints = get_type_hints(schema)
    except (NameError, TypeError):  # pragma: no cover - defensive fallback
        type_hints = {field.name: field.type for field in fields(schema)}

    kwargs: dict[str, Any] = {}
    for field in fields(schema):
        if field.name not in data:
            continue
        value = data[field.name]
        kwargs[field.name] = _convert_value_by_type(value, type_hints.get(field.name, field.type))

    return schema(**kwargs)


def _convert_value_by_type(value: Any, field_type: Any) -> Any:
    origin = get_origin(field_type)

    if origin in {list, tuple}:
        if value is None:
            return [] if origin is list else tuple()
        item_type = get_args(field_type)[0] if get_args(field_type) else Any
        if is_dataclass(item_type):
            converted = [convert_to_dataclass(item, item_type) for item in value]
        else:
            converted = list(value)
        return converted if origin is list else tuple(converted)

    if origin in {dict, Mapping}:
        if value is None:
            return {}
        key_type, value_type = get_args(field_type) if get_args(field_type) else (Any, Any)
        if is_dataclass(value_type):
            return {key: convert_to_dataclass(val, value_type) for key, val in value.items()}
        return dict(value)

    if origin in {types.UnionType, Union}:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]  # noqa: E721
        if not args:
            return value
        target_type = args[0]
        if value is None:
            return None
        return _convert_value_by_type(value, target_type)

    if is_dataclass(field_type):
        return convert_to_dataclass(value, field_type)

    return value


__all__ = ["convert_to_dataclass"]
