"""Tests for config utilities."""

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ReadonlyConfigError

from simplexity.exceptions import ConfigValidationError
from simplexity.utils.config_utils import (
    TARGET,
    dynamic_resolve,
    filter_instance_keys,
    get_config,
    get_instance_keys,
    typed_instantiate,
)


def test_get_instance_keys_no_targets() -> None:
    """Test getting instance keys when there are no targets."""
    cfg = DictConfig(
        {
            "int_item": 1,
            "str_itme": "str",
            "list_item": ListConfig([1, 2, 3]),
            "dict_item": DictConfig({"a": 1, "b": 2, "c": 3}),
        }
    )
    assert not get_instance_keys(cfg)


def test_get_instance_keys_target_config() -> None:
    """Test getting instance keys when there is a target config."""
    cfg = DictConfig(
        {
            TARGET: "some_callable",
            "args": ListConfig(["arg1", "arg2", "arg3"]),
            "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
        }
    )
    assert not get_instance_keys(cfg)


def test_get_instance_keys_top_level() -> None:
    """Test getting instance keys when there are top level targets."""
    cfg = DictConfig(
        {
            "instance1": DictConfig(
                {
                    TARGET: "some_callable",
                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                }
            ),
            "instance2": DictConfig(
                {
                    TARGET: "some_callable",
                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                }
            ),
            "instance3": DictConfig(
                {
                    TARGET: "some_callable",
                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                }
            ),
        }
    )
    assert set(get_instance_keys(cfg)) == set(["instance1", "instance2", "instance3"])


def test_get_instance_keys_multi_level() -> None:
    """Test getting instance keys when there are multi level targets."""
    cfg = DictConfig(
        {
            "instance1": DictConfig(
                {
                    TARGET: "some_callable",
                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                }
            ),
            "level2": DictConfig(
                {
                    "instance2": DictConfig(
                        {
                            TARGET: "some_callable",
                            "args": ListConfig(["arg1", "arg2", "arg3"]),
                            "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                        }
                    ),
                    "level3": DictConfig(
                        {
                            "instance3": DictConfig(
                                {
                                    TARGET: "some_callable",
                                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )
    assert set(get_instance_keys(cfg)) == set(
        [
            "instance1",
            "level2.instance2",
            "level2.level3.instance3",
        ]
    )


def test_get_instance_keys_ignore_nested() -> None:
    """Test getting instance keys when there are nested targets."""
    cfg = DictConfig(
        {
            "instance1": DictConfig(
                {
                    TARGET: "some_callable",
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                    "instance2": DictConfig(
                        {
                            TARGET: "some_callable",
                            "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                            "instance3": DictConfig(
                                {
                                    TARGET: "some_callable",
                                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )
    assert get_instance_keys(cfg, nested=False) == ["instance1"]


def test_get_instance_keys_include_nested() -> None:
    """Test getting instance keys when there are nested targets and include nested is True."""
    cfg = DictConfig(
        {
            "instance1": DictConfig(
                {
                    TARGET: "some_callable",
                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                    "instance2": DictConfig(
                        {
                            TARGET: "some_callable",
                            "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                            "instance3": DictConfig(
                                {
                                    TARGET: "some_callable",
                                    "args": ListConfig(["arg1", "arg2", "arg3"]),
                                    "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )
    assert set(get_instance_keys(cfg, nested=True)) == set(
        ["instance1", "instance1.instance2", "instance1.instance2.instance3"]
    )


def test_filter_instance_keys() -> None:
    """Test filtering instance keys."""
    cfg = DictConfig(
        {
            "instance1": DictConfig(
                {
                    TARGET: "some_callable",
                }
            ),
            "instance2": DictConfig(
                {
                    TARGET: "some_other_callable",
                }
            ),
            "sub_config": DictConfig(
                {
                    "instance3": DictConfig(
                        {
                            TARGET: "some_callable",
                        }
                    ),
                    "instance4": DictConfig(
                        {
                            TARGET: "some_other_callable",
                        }
                    ),
                }
            ),
        }
    )

    def is_some_callable(target: str) -> bool:
        return target == "some_callable"

    all_instance_keys = ["instance1", "instance2", "sub_config.instance3", "sub_config.instance4"]
    assert filter_instance_keys(cfg, all_instance_keys, is_some_callable) == ["instance1", "sub_config.instance3"]


def test_filter_instance_keys_with_validation() -> None:
    """Test filtering instance keys with validation."""
    cfg = DictConfig(
        {
            "component1": DictConfig(
                {
                    "instance": DictConfig(
                        {
                            TARGET: "some_callable",
                        }
                    ),
                    "is_valid": False,
                }
            ),
            "component2": DictConfig(
                {
                    "instance": DictConfig(
                        {
                            TARGET: "some_callable",
                        }
                    ),
                    "is_valid": True,
                }
            ),
            "sub_config": DictConfig(
                {
                    "component3": DictConfig(
                        {
                            "instance": DictConfig(
                                {
                                    TARGET: "some_callable",
                                }
                            ),
                            "is_valid": False,
                        }
                    ),
                    "component4": DictConfig(
                        {
                            "instance": DictConfig(
                                {
                                    TARGET: "some_callable",
                                }
                            ),
                            "is_valid": True,
                        }
                    ),
                }
            ),
        }
    )

    def is_anything(target: str) -> bool:
        return len(target.strip()) > 0

    def is_valid(cfg: DictConfig) -> None:
        if not cfg.get("is_valid", False):
            raise ConfigValidationError("is_valid must be True")

    all_instance_keys = [
        "component1.instance",
        "component2.instance",
        "sub_config.component3.instance",
        "sub_config.component4.instance",
    ]
    assert filter_instance_keys(cfg, all_instance_keys, is_anything, is_valid) == [
        "component2.instance",
        "sub_config.component4.instance",
    ]


def test_get_config_from_kwargs() -> None:
    """Test getting config from kwargs."""
    cfg = DictConfig({"key": "value"})
    assert get_config((), {"cfg": cfg}) == cfg


def test_get_config_from_args() -> None:
    """Test getting config from args."""
    cfg = DictConfig({"key": "value"})
    assert get_config((cfg,), {}) == cfg


def test_get_config_from_no_args() -> None:
    """Test getting config from no args."""
    with pytest.raises(ValueError, match="No config found in arguments or kwargs."):
        get_config((), {})


def test_dynamic_resolve() -> None:
    """Test dynamic resolve."""

    @dynamic_resolve
    def mutate_cfg(cfg: DictConfig) -> None:
        cfg.key = "new_value"
        cfg.new_key = "new_value"

    cfg = DictConfig({"key": "value"})
    mutate_cfg(cfg)
    assert cfg.key == "new_value"
    assert cfg.new_key == "new_value"
    assert OmegaConf.is_struct(cfg)
    assert OmegaConf.is_readonly(cfg)
    with pytest.raises(ReadonlyConfigError, match="readonly|Readonly|read-only"):
        cfg.key = "newer_value"


def test_typed_instantiate() -> None:
    """Test typed instantiate with built-in str type."""
    cfg = DictConfig({"_target_": "builtins.str", "object": 42})
    obj = typed_instantiate(cfg, str)
    assert obj == "42"
    assert isinstance(obj, str)


def test_typed_instantiate_with_kwargs() -> None:
    """Test typed instantiate with keyword arguments."""
    cfg = DictConfig({"_target_": "builtins.str"})
    obj = typed_instantiate(cfg, str, object=42)
    assert obj == "42"
    assert isinstance(obj, str)


def test_typed_instantiate_type_error() -> None:
    """Test typed instantiate raises AssertionError when type doesn't match."""
    cfg = DictConfig({"_target_": "builtins.str", "object": 42})
    with pytest.raises(AssertionError):
        typed_instantiate(cfg, int)


def test_typed_instantiate_with_string_expected_type() -> None:
    """Test typed instantiate with expected type as string."""
    cfg = DictConfig({"_target_": "builtins.str", "object": 42})
    obj = typed_instantiate(cfg, "builtins.str")
    assert obj == "42"
    assert isinstance(obj, str)
