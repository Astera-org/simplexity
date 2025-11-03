from omegaconf import DictConfig, ListConfig

from simplexity.utils.config_utils import TARGET, get_instance_keys


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
