"""Load and merge configs from MLflow runs.

This module implements the mlflow_defaults functionality, allowing configs to be
composed from parts or entire configs that have been logged as MLflow artifacts
from previous runs. The design is patterned after Hydra's defaults list.

See LOAD_SUBCONFIGS.md for the full specification.
"""

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

import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from simplexity.exceptions import ConfigValidationError
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.structured_configs.mlflow import resolve_mlflow_config, validate_mlflow_config
from simplexity.utils.config_utils import dynamic_resolve

DEFAULT_ARTIFACT_NAME = "config"

# TARGET(@PACKAGE)?
# [optional|override]? TARGET(@PACKAGE)? : OPTION
FLAGS_STR = r"(?P<flags>(?:(?:optional|override)\s+)*)"
MLFLOW_CONFIG_ENTRY_STR = r"(?P<target>[\w\.]+)(?:@(?P<package>[\w\.]+))?"
OPTION_STR = r"(?P<option>.*?)"
MLFLOW_DEFAULT_ITEM_PATTERN = re.compile(rf"^{FLAGS_STR}(?:{MLFLOW_CONFIG_ENTRY_STR})?(?:\s*:\s*{OPTION_STR})?$")


class _ParsedEntry(NamedTuple):
    """Parsed MLflow default item."""

    optional: bool
    override: bool
    target: str
    package: str
    artifact_path: str | None
    select_path: str | None


def _parse_option(option: str) -> tuple[str, str | None]:
    """Parse artifact path and select path from option string."""
    if "#" in option:
        artifact_part, select_part = option.split("#", 1)
        artifact_path = artifact_part.strip() or DEFAULT_ARTIFACT_NAME
        select_path = select_part.strip() or None
        return artifact_path, select_path

    if "/" in option:
        artifact_path = option.strip()
        select_path = None
        return artifact_path, select_path

    artifact_path = DEFAULT_ARTIFACT_NAME
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
    override = "override" in flags_str

    target = groups["target"]
    if not target:
        raise ValueError(
            f"Invalid MLflow default entry: {item}. Must be in format "
            "'TARGET(@PACKAGE)? | [optional|override]? TARGET(@PACKAGE)?: OPTION | _self_'"
        )

    package = groups.get("package")
    option = groups.get("option")

    # Check for null keyword first, before special case handling
    if option == "null":
        package = package or "."
        return _ParsedEntry(optional, override, target, package, None, None)

    # Special case: TARGET: VALUE (no @) where VALUE doesn't contain / or #
    # Treat VALUE as both PACKAGE and SELECT_PATH
    if package is None and option is not None and "/" not in option and "#" not in option:
        package = option.rsplit(".", 1)[1] if "." in option else option
        artifact_path = DEFAULT_ARTIFACT_NAME
        select_path = option
        return _ParsedEntry(optional, override, target, package, artifact_path, select_path)

    package = package or "."

    artifact_path, select_path = _parse_option(option or "")
    return _ParsedEntry(optional, override, target, package, artifact_path, select_path)


def _get_target_config(cfg: DictConfig, parsed_entry: _ParsedEntry) -> Any | None:
    """Download and load config from MLflow run.

    Args:
        cfg: The current config containing MLflow config nodes.
        parsed_entry: The parsed entry specifying target, package, and paths.

    Returns:
        The loaded config (or selected subconfig), or None if loading failed
        and the entry is optional.
    """
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
        resolve_mlflow_config(target_node, create_if_missing=False)
    except (ValueError, RuntimeError) as e:
        SIMPLEXITY_LOGGER.warning("Error resolving MLflow config: %s", e)
        return None

    tracking_uri: str | None = target_node.get("tracking_uri")
    run_id: str = target_node.get("run_id")
    artifact_path = parsed_entry.artifact_path
    if not artifact_path.endswith((".yaml", ".yml")):
        artifact_path = f"{artifact_path}.yaml"
    client = MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_path = client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=tmp_dir)
        except (MlflowException, RestException, OSError, FileNotFoundError) as e:
            SIMPLEXITY_LOGGER.warning("Failed to download artifact from MLflow '%s': %s", parsed_entry.target, e)
            return None

        try:
            loaded_config = OmegaConf.load(local_path)
        except (OSError, FileNotFoundError, ValueError, yaml.YAMLError) as e:
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


def _normalize_item(item: str | DictConfig) -> str:
    """Normalize an item to a string format.

    If item is a DictConfig with a single key-value pair:
    - If value is "config" (the default artifact), treat as just the key (no option)
    - If value is None (Python None, from YAML null), convert to "key: null" string
    - Otherwise, convert to "key: value" format
    Otherwise, convert to string.
    """
    if isinstance(item, DictConfig):
        keys = list(item.keys())
        if len(keys) == 1:
            key = keys[0]
            value = item[key]
            # If value is the default artifact name, treat as simple CONFIG entry
            if value == DEFAULT_ARTIFACT_NAME:
                return str(key)
            # If value is None (from YAML null), convert to "null" string
            # IMPORTANT: Check for None BEFORE f-string formatting to avoid converting
            # Python None to the string "None" instead of "null"
            if value is None:
                return f"{key}: null"
            # Format the value - this would convert None to "None" if the check above is bypassed
            return f"{key}: {value}"
        # Multiple keys - convert entire dict to string representation
        return str(item)
    return str(item)


def _resolve_mlflow_configs_recursive(cfg: DictConfig) -> None:
    """Recursively find and resolve all MLflow configs in the config.

    This is needed when _self_ is processed and MLflow configs are merged unresolved.
    MLflow configs that are not valid or can't be resolved are silently skipped.

    Args:
        cfg: The DictConfig to search for MLflow configs.
    """
    if not isinstance(cfg, DictConfig):
        return

    # Only try to validate/resolve if this looks like an MLflow config
    # Check for key identifying fields (not just optional fields like registry_uri)
    mlflow_identifying_fields = {"experiment_id", "experiment_name", "run_id", "run_name"}
    has_mlflow_fields = any(field in cfg for field in mlflow_identifying_fields)

    if has_mlflow_fields:
        # Try to validate and resolve the current node as an MLflow config
        try:
            validate_mlflow_config(cfg)
            # If validation passes, this looks like an MLflow config, try to resolve it
            resolve_mlflow_config(cfg, create_if_missing=False)
        except (ConfigValidationError, ValueError, RuntimeError):
            # Not an MLflow config or can't be resolved, continue to check nested configs
            pass

    # Recursively process all nested DictConfig values
    for key in cfg:
        try:
            value = cfg[key]
        except MissingMandatoryValue:
            # Skip keys that can't be accessed (e.g., missing mandatory values)
            continue

        if isinstance(value, DictConfig):
            _resolve_mlflow_configs_recursive(value)
        elif isinstance(value, ListConfig):
            # Also check ListConfig items that might be DictConfigs
            for item in value:
                if isinstance(item, DictConfig):
                    _resolve_mlflow_configs_recursive(item)


def _process_entry(cfg: DictConfig, accumulator: DictConfig, item: str) -> DictConfig:
    """Process a single MLflow default item.

    Handles both "_self_" entries (merging the original config) and MLflow entries
    (downloading and merging configs from MLflow runs). Uses deep merge semantics
    matching Hydra's defaults list behavior, unless the override flag is specified.

    When override is True, the loaded config completely replaces the value at the
    package path instead of merging with existing content.

    Args:
        cfg: The original config being processed.
        accumulator: The accumulated merged config so far.
        item: The entry to process (either "_self_" or an MLflow entry string).

    Returns:
        The updated accumulator with the entry merged in or replaced (if override).
    """
    if item == "_self_":
        # Standard merge semantics: last entry wins (matches Hydra behavior)
        # OmegaConf.merge does deep merge (nested dicts get merged), which matches
        # how Hydra composes configs with defaults list
        result = cast(DictConfig, OmegaConf.merge(accumulator, cfg))
        # Resolve any MLflow configs that may have been merged unresolved
        _resolve_mlflow_configs_recursive(result)
        return result

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
        if parsed_entry.override:
            # Override at root: replace entire accumulator with loaded config
            return loaded_config
        return cast(DictConfig, OmegaConf.merge(accumulator, loaded_config))

    # When merging at a package (not root)
    if parsed_entry.override:
        # Override: completely replace the value at the package path
        # Traverse the path and directly assign the value to replace any existing content
        package_parts = parsed_entry.package.split(".")
        target = accumulator
        # Navigate to the parent of the target key
        for part in package_parts[:-1]:
            if part not in target:
                target[part] = OmegaConf.create()
            target = target[part]
        # Directly assign the new value, completely replacing any existing value
        target[package_parts[-1]] = loaded_config
        return accumulator

    # Standard merge: use deep merge semantics (matches Hydra behavior)
    # This ensures that MLflow content merges with any existing content at that package path,
    # preserving non-conflicting keys and deeply merging nested dictionaries
    package_conf = OmegaConf.create()
    OmegaConf.update(package_conf, parsed_entry.package, loaded_config)
    return cast(DictConfig, OmegaConf.merge(accumulator, package_conf))


@dynamic_resolve
def load_mlflow_defaults(cfg: DictConfig) -> DictConfig:
    """Load and merge configs from MLflow runs based on mlflow_defaults list.

    Processes the mlflow_defaults list in the config, downloading artifacts from
    MLflow runs and merging them according to the specified package paths and
    select paths. If "_self_" is not present, it is appended to the end.

    Args:
        cfg: The config containing mlflow_defaults list and MLflow config nodes.

    Returns:
        A new config with MLflow defaults merged in. The mlflow_defaults key is
        removed from the result.

    See Also:
        LOAD_SUBCONFIGS.md for the full specification and examples.
    """
    mlflow_defaults = cfg.get("mlflow_defaults")
    if mlflow_defaults is None:
        return cfg

    mlflow_defaults_list = [mlflow_defaults] if isinstance(mlflow_defaults, str) else list(mlflow_defaults)

    # Create a copy to avoid mutating the input config
    mlflow_defaults_copy = cast(ListConfig, OmegaConf.create(mlflow_defaults_list))
    if "_self_" not in mlflow_defaults_copy:
        mlflow_defaults_copy.append("_self_")

    accumulator = cast(DictConfig, OmegaConf.create())

    for item in mlflow_defaults_copy:
        normalized_item = _normalize_item(item)
        accumulator = _process_entry(cfg, accumulator, normalized_item)

    # Remove mlflow_defaults key if it exists (it may not exist if override at root replaced everything)
    if "mlflow_defaults" in accumulator:
        del accumulator["mlflow_defaults"]

    return accumulator
