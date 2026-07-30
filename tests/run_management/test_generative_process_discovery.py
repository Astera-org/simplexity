"""Tests for discovering and instantiating generative processes, including foreign ones.

"Foreign" means a process whose implementation lives outside `simplexity.generative_processes` —
typically vendored from generators into the consumer's project — so it cannot be recognized from
its import path.
"""

import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.run_management.run_management import (
    _instantiate_generative_process,
    _setup_generative_processes,
)
from simplexity.utils.config_utils import get_instance_keys

SIMPLEXITY_TARGET = "simplexity.generative_processes.builder.build_hidden_markov_model"
FOREIGN_TARGET = "tests.vendored_process.adapter.build_vendored_mess3"
FOREIGN_INSTANCE = {"_target_": FOREIGN_TARGET, "x": 0.15, "a": 0.6}
SIMPLEXITY_INSTANCE = {"_target_": SIMPLEXITY_TARGET, "process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}


def _process_cfg(instance: dict, *, declare: bool) -> DictConfig:
    section: dict = {"name": "process", "instance": instance}
    if declare:
        section["component"] = "generative_process"
    section |= {"base_vocab_size": "???", "bos_token": "???", "eos_token": None, "vocab_size": "???"}
    return OmegaConf.create({"generative_process": section})


def _setup(cfg: DictConfig):
    return _setup_generative_processes(cfg, get_instance_keys(cfg))


def test_declared_foreign_process_is_instantiated() -> None:
    """A declared foreign target is discovered even though its namespace is not simplexity."""
    processes = _setup(_process_cfg(FOREIGN_INSTANCE, declare=True))
    assert processes is not None
    assert list(processes) == ["generative_process.instance"]
    assert type(processes["generative_process.instance"]).__name__ == "VendoredGhmmProcess"


def test_declared_foreign_process_resolves_vocabulary_fields() -> None:
    """Resolution reads the instantiated process, so it works for foreign processes too."""
    cfg = _process_cfg(FOREIGN_INSTANCE, declare=True)
    _setup(cfg)
    assert cfg.generative_process.base_vocab_size == 3
    assert cfg.generative_process.bos_token == 3
    assert cfg.generative_process.vocab_size == 4


def test_simplexity_process_needs_no_declaration() -> None:
    """The namespace fast path keeps every existing config working unchanged."""
    processes = _setup(_process_cfg(SIMPLEXITY_INSTANCE, declare=False))
    assert processes is not None
    assert list(processes) == ["generative_process.instance"]


def test_declaring_a_simplexity_process_is_harmless() -> None:
    """Declaration is additive, so configs may adopt it before the namespace path retires."""
    processes = _setup(_process_cfg(SIMPLEXITY_INSTANCE, declare=True))
    assert processes is not None
    assert list(processes) == ["generative_process.instance"]


def test_undeclared_foreign_process_is_not_discovered() -> None:
    """Without a declaration, a foreign target is indistinguishable from any other instance."""
    assert _setup(_process_cfg(FOREIGN_INSTANCE, declare=False)) is None


def test_config_without_a_process_yields_none() -> None:
    """A config that genuinely configures no process is still not an error."""
    cfg = OmegaConf.create({"optimizer": {"instance": {"_target_": "torch.optim.Adam", "lr": 0.01}}})
    assert _setup(cfg) is None


def test_declared_process_with_an_invalid_config_raises() -> None:
    """A section that declares a process must not be dropped silently.

    Before this behaviour existed, an unusable declaration left `generative_processes` as None,
    which is indistinguishable from a config that configures no process at all.
    """
    cfg = _process_cfg({"_target_": ""}, declare=True)
    with pytest.raises(ConfigValidationError, match="declares generative processes at"):
        _setup(cfg)


def test_declared_process_error_names_the_offending_key() -> None:
    cfg = _process_cfg({"_target_": ""}, declare=True)
    with pytest.raises(ConfigValidationError, match="generative_process.instance"):
        _setup(cfg)


def test_undeclared_invalid_foreign_config_is_skipped_not_raised() -> None:
    """Unclaimed sections stay skippable; only declared intent turns a skip into an error."""
    assert _setup(_process_cfg({"_target_": "torch.optim.Adam"}, declare=False)) is None


def test_instantiating_a_non_process_raises_naming_missing_members() -> None:
    """A declared target that instantiates something else fails with an actionable message."""
    cfg = _process_cfg({"_target_": "builtins.dict"}, declare=True)
    with pytest.raises(ConfigValidationError, match="which is not a generative process: missing"):
        _instantiate_generative_process(cfg, "generative_process.instance")


def test_instantiating_a_non_process_lists_the_absent_operations() -> None:
    cfg = _process_cfg({"_target_": "builtins.dict"}, declare=True)
    with pytest.raises(ConfigValidationError, match="emit_observation"):
        _instantiate_generative_process(cfg, "generative_process.instance")


def test_instantiating_a_missing_instance_key_raises_key_error() -> None:
    cfg = _process_cfg(FOREIGN_INSTANCE, declare=True)
    with pytest.raises(KeyError):
        _instantiate_generative_process(cfg, "generative_process.absent")
