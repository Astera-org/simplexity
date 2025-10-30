from omegaconf import DictConfig, ListConfig

from simplexity.utils.hydra import TARGET, get_targets


def test_get_targets_no_targets() -> None:
    cfg = DictConfig(
        {
            "int_item": 1,
            "str_itme": "str",
            "list_item": ListConfig([1, 2, 3]),
            "dict_item": DictConfig({"a": 1, "b": 2, "c": 3}),
        }
    )
    assert get_targets(cfg) == []


def test_get_targets_target_config() -> None:
    cfg = DictConfig(
        {
            TARGET: "some_callable",
            "args": ListConfig(["arg1", "arg2", "arg3"]),
            "kwargs": DictConfig({"arg4": 4, "arg5": 5, "arg6": 6}),
        }
    )
    assert get_targets(cfg) == []


def test_get_targets_top_level() -> None:
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
    assert set(get_targets(cfg)) == set(["instance1", "instance2", "instance3"])


def test_get_targets_multi_level() -> None:
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
    assert set(get_targets(cfg)) == set(["instance1", "level2.instance2", "level2.level3.instance3"])


def test_get_targets_ignore_nested() -> None:
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
    assert get_targets(cfg, nested=False) == ["instance1"]


    def test_get_targets_ignore_nested() -> None:
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
    assert set(get_targets(cfg, nested=True)) == set(["instance1", "instance1.instance2", "instance1.instance2.instance3"])
