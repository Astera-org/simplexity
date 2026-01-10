"""Config utilities."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import re
import tempfile
from typing import Any, NamedTuple, cast

from mlflow import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.structured_configs.mlflow import resolve_mlflow_config, validate_mlflow_config
from simplexity.utils.config_utils import dynamic_resolve

# TARGET(@PACKAGE)?
# [optional|override]? TARGET(@PACKAGE)? : OPTION
FLAGS_STR = r"(?P<flags>(?:(?:optional|override)\s+)*)"
MLFLOW_CONFIG_ENTRY_STR = r"(?P<target>[\w\.]+)(?:@(?P<package>[\w\.]+))?"
OPTION_STR = r"(?P<option>.*?)"
MLFLOW_DEFAULT_ITEM_PATTERN = re.compile(rf"^{FLAGS_STR}(?:{MLFLOW_CONFIG_ENTRY_STR})?(?:\s*:\s*{OPTION_STR})?$")


class _ParsedEntry(NamedTuple):
    """Parsed MLflow default item."""

    optional: bool
    target: str
    package: str
    artifact_path: str | None
    select_path: str | None


def _parse_option(option: str) -> tuple[str, str | None]:
    """Parse artifact path and select path from option string."""
    if "#" in option:
        artifact_part, select_part = option.split("#", 1)
        artifact_path = artifact_part.strip() or "config"
        select_path = select_part.strip() or None
        return artifact_path, select_path

    if "/" in option:
        artifact_path = option.strip()
        select_path = None
        return artifact_path, select_path

    artifact_path = "config"
    select_path = option.strip() or None
    return artifact_path, select_path


def _parse_entry(item: str) -> _ParsedEntry:
    """Parse a single MLflow default item."""
    match = MLFLOW_DEFAULT_ITEM_PATTERN.match(item)
    if not match:
        raise ValueError(
            f"Invalid MLflow default entry: {item}. Must be in format '[optional|override]* TARGET(@PACKAGE)?: OPTION'"
        )

    groups = match.groupdict()
    flags_str = groups.get("flags", "")
    optional = "optional" in flags_str

    target = groups["target"]
    if not target:
        raise ValueError(
            f"Invalid MLflow default entry: {item}. Must be in format "
            "'TARGET(@PACKAGE)? | [optional|override]? TARGET(@PACKAGE)?: OPTION | _self_'"
        )

    package = groups.get("package") or "."
    option = groups.get("option")

    if option == "null":
        return _ParsedEntry(optional, target, package, None, None)

    artifact_path, select_path = _parse_option(option or "")
    return _ParsedEntry(optional, target, package, artifact_path, select_path)


def _get_target_config(cfg: DictConfig, parsed_entry: _ParsedEntry) -> Any | None:
    if parsed_entry.artifact_path is None:
        SIMPLEXITY_LOGGER.warning("Config is mandatory but OPTION is null for entry: %s", parsed_entry.target)
        return None

    target_node: DictConfig | None = OmegaConf.select(cfg, parsed_entry.target)
    if target_node is None:
        SIMPLEXITY_LOGGER.warning("Target node '%s' not found in config", parsed_entry.target)
        return None

    try:
        validate_mlflow_config(target_node)
    except ConfigValidationError as e:
        SIMPLEXITY_LOGGER.warning("Error validating MLflow config: %s", e)
        return None

    try:
        resolve_mlflow_config(target_node)
    except ValueError as e:
        SIMPLEXITY_LOGGER.warning("Error resolving MLflow config: %s", e)
        return None

    tracking_uri: str | None = target_node.get("tracking_uri")
    run_id: str = target_node.get("run_id")
    client = MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_path = client.download_artifacts(run_id=run_id, path=parsed_entry.artifact_path, dst_path=tmp_dir)
        except Exception as e:
            SIMPLEXITY_LOGGER.warning("Failed to download artifact from MLflow '%s': %s", parsed_entry.target, e)
            return None

        try:
            loaded_config = OmegaConf.load(local_path)
        except Exception as e:
            SIMPLEXITY_LOGGER.warning("Failed to load MLflow default '%s': %s", parsed_entry.target, e)
            return None

    if parsed_entry.select_path is None:
        return loaded_config

    selected_config = OmegaConf.select(loaded_config, parsed_entry.select_path)
    if selected_config is None:
        SIMPLEXITY_LOGGER.warning(
            "Selected path '%s' not found in artifact '%s'", parsed_entry.select_path, parsed_entry.artifact_path
        )
        return None

    return selected_config


def _process_entry(cfg: DictConfig, accumulator: DictConfig, item: str) -> DictConfig:
    """Process a single MLflow default item."""
    if item == "_self_":
        return cast(DictConfig, OmegaConf.merge(accumulator, cfg))

    parsed_entry = _parse_entry(item)

    loaded_config = _get_target_config(cfg, parsed_entry)

    if loaded_config is None:
        if parsed_entry.optional:
            return accumulator
        raise ValueError(f"Target config not found for entry: {item}")

    if parsed_entry.package == ".":
        if not isinstance(loaded_config, DictConfig):
            if parsed_entry.optional:
                return accumulator
            raise ValueError(f"Target config not found for entry: {item}")
        return cast(DictConfig, OmegaConf.merge(accumulator, loaded_config))

    package_conf = OmegaConf.create()
    OmegaConf.update(package_conf, parsed_entry.package, loaded_config)
    return cast(DictConfig, OmegaConf.merge(accumulator, package_conf))


@dynamic_resolve
def load_mlflow_defaults(cfg: DictConfig) -> DictConfig:
    """Load defaults from MLflow runs."""
    mlflow_defaults: ListConfig | None = cfg.get("mlflow_defaults")
    if mlflow_defaults is None:
        return cfg

    if "_self_" not in mlflow_defaults:
        mlflow_defaults.append("_self_")

    accumulator = OmegaConf.create()

    for item in mlflow_defaults:
        accumulator = _process_entry(cfg, accumulator, item)

    return accumulator
