# Loading Subconfigs from MLflow runs

## Goal:

The goal is to be able to compose a config taking parts/entire configs that have been logged as MLflow artifacts from previous runs

## Design:

The proposed design is patterned after the Hydra [defaults list](https://hydra.cc/docs/advanced/defaults_list/):

```
mlflow_defaults:
  (- MLFLOW_ENTRY)*

MLFLOW_ENTRY      : CONFIG | OPTION_CONFIG | _self_
CONFIG            : TARGET(@PACKAGE)?
OPTION_CONFIG     : [optional|override]? TARGET(@PACKAGE)?: OPTION
OPTION            : ARTIFACT_PATH | SELECT_PATH | ARTIFACT_PATH#SELECT_PATH | null
```

- `TARGET`: OmegaConf dot path within current config pointing to the [MLFlowConfig](https://github.com/Astera-org/simplexity/blob/657aff777cd7d7a6edfadd1f846bbf62028caeec/simplexity/structured_configs/mlflow.py#L21) node
- `PACKAGE`: OmegaConf dot path where to place merged content (default: ".", i.e., root)
- `ARTIFACT_PATH`: path within the MLflow run’s artifact dir to the source YAML without system extension (default: "config").
  - If the path contains `/`, it is treated as an `ARTIFACT_PATH`.
  - If the path does not contain `/`, it is ambiguous and defaults to `SELECT_PATH`. To force it to be an `ARTIFACT_PATH`, append `#` (e.g., `my_artifact#`).
- `SELECT_PATH`: OmegaConf dot path within the source YAML to the subconfig to import (default: root).
- `_self_`, `optional`, and `override` function the same as their Hydra equivalents.
  - `_self_` determines the composition order. If missing, it is always appended to the end. If `_self_` is explicitly included, it is processed in the specified order.
  - Standard merge semantics apply: **whatever comes last wins**, whether it's `_self_` or an MLflow entry.
    - If `_self_` is first: MLflow entries merged after `_self_` will override the original config.
    - If `_self_` is last: Original config merged after MLflow entries will override MLflow content.
    - To override everything from `_self_`, explicitly include `_self_` first, then use `override` at root.
  - `optional` suppresses errors if the artifact or selected subconfig is missing.
  - `override` causes the loaded config to completely replace the value at the package path instead of merging. When override is used, any existing content at that path is completely replaced with the loaded content (no deep merge).

### Merge Semantics for Overlapping Keys

When multiple entries in `mlflow_defaults` (or `_self_`) have overlapping keys, the merge behavior follows Hydra's deep merge semantics:

1. **Conflicting keys (same key path)**: The last entry wins. The value from the later entry completely replaces the value from earlier entries.

   ```yaml
   # Entry 1: other_section.foo = "first"
   # Entry 2: other_section.foo = "second"
   # Result: other_section.foo = "second" (Entry 2 wins)
   ```

2. **Non-conflicting keys at the same level**: Both keys are preserved. Nested dictionaries are merged deeply.

   ```yaml
   # Entry 1: other_section.key = 3, other_section.dict.subkey = 30
   # Entry 2: other_section.foo = "bar"
   # Result: other_section.key = 3, other_section.dict.subkey = 30, other_section.foo = "bar"
   # (All keys preserved, nested dicts merged)
   ```

3. **Partially overlapping nested structures**: Deep merge preserves non-conflicting nested keys.
   ```yaml
   # Entry 1: other_section.dict.key1 = "value1", other_section.dict.key2 = "value2"
   # Entry 2: other_section.dict.key2 = "new_value2", other_section.dict.key3 = "value3"
   # Result:
   #   other_section.dict.key1 = "value1" (preserved from Entry 1)
   #   other_section.dict.key2 = "new_value2" (Entry 2 wins for conflicting key)
   #   other_section.dict.key3 = "value3" (added from Entry 2)
   ```

This matches Hydra's defaults list behavior: entries are merged in order using `OmegaConf.merge()`, which performs deep merging of nested dictionaries.

### OPTION Parsing Logic

To resolve ambiguity in `OPTION`:

1. If `OPTION` contains `#`:
   - Split into `ARTIFACT_PART` and `SELECT_PART`.
   - `ARTIFACT_PATH` = `ARTIFACT_PART` (if empty, defaults to "config").
   - `SELECT_PATH` = `SELECT_PART` (if empty, defaults to root).
2. Else (no `#`):
   - If `OPTION` contains `/`:
     - `ARTIFACT_PATH` = `OPTION`.
     - `SELECT_PATH` = root.
   - Else:
     - `ARTIFACT_PATH` = "config".
     - `SELECT_PATH` = `OPTION`.

## Examples:

### load entire `config.yaml` and merge at root

```yaml
defaults:
  - mlflow@load_source: previous_run
mlflow_defaults:
  - load_source
```

### load `configs/model.yaml` and merge as old_model

```yaml
defaults:
  - mlflow@load_source: previous_run
mlflow_defaults:
  # 'configs/model' contains '/', so it is treated as ARTIFACT_PATH
  - load_source@old_model: configs/model
```

### load `persistence` subconfig from `config.yaml` if it exists

```yaml
defaults:
  - mlflow@load_source: previous_run
mlflow_defaults:
  # 'persistence' has no '/' or '#', so it is treated as SELECT_PATH from default artifact 'config'
  - optional load_source@persistence: persistence
```

### override `train.optimizer` with subconfig from` train.yaml`

```yaml
defaults:
  - mlflow@load_source: previous_run
  - train: smoke
mlflow_defaults:
  # usage of '#' explicit defines ARTIFACT_PATH (train) and SELECT_PATH (optimizer)
  # override flag causes complete replacement of train.optimizer (no merge)
  - override load_source@train.optimizer: train#optimizer
```

**Note**: With `override`, the entire `train.optimizer` section is replaced with the content from the artifact. Any existing keys in `train.optimizer` that are not in the artifact will be removed.

### load `model` and `generative_process` from previous run

```yaml
defaults:
  - mlflow@load_source: previous_run
mlflow_defaults:
  - load_source@model: model
  - load_source@generative_process: generative_process
```

### load models from multiple runs

```yaml
defaults:
  - model@current_model: transformer
  - mlflow@load_source_1: previous_run_1
  - mlflow@load_source_2: previous_run_2
mlflow_defaults:
  - load_source_1@old_model_1: model
  - load_source_2@old_model_2: model
```

### Load a top-level artifact `custom.yaml` at root

```yaml
defaults:
  - mlflow@load_source: previous_run
mlflow_defaults:
  # Use '#' to indicate 'custom' is the artifact, and select root
  - load_source@custom_section: custom#
```

### Merge behavior with overlapping keys

Deep merge semantics apply consistently everywhere, matching Hydra's defaults list behavior:

1. **Loading at root (no package specified)**: Deep merge - multiple entries merge together, preserving non-conflicting keys.

2. **Loading at a specific package path (e.g., `@other_section`)**: Deep merge - content merges with any existing content at that package path, preserving non-conflicting keys.

   ```yaml
   defaults:
     - mlflow@load_source: previous_run
   other_section:
     foo: bar
   mlflow_defaults:
     - _self_
     - load_source@other_section: config#
   # Result: other_section contains both foo: bar (from _self_)
   # and content from config (merged together)
   ```

3. **Multiple entries at the same package path**: Deep merge preserves non-conflicting keys:
   ```yaml
   mlflow_defaults:
     - load_source_1@other_section: config1# # Loads at other_section
     - load_source_2@other_section: config2# # Merges with config1's other_section
   ```

**Example of multiple entries with overlapping keys:**

```yaml
defaults:
  - mlflow@load_source_1: previous_run_1
  - mlflow@load_source_2: previous_run_2
mlflow_defaults:
  # Load config from run 1
  - load_source_1@other_section: config1#
  # Load config from run 2 (will merge with run 1's other_section)
  - load_source_2@other_section: config2#
```

If `config1.yaml` contains:

```yaml
other_section:
  key: 3
  dict:
    subkey: 30
```

And `config2.yaml` contains:

```yaml
other_section:
  foo: bar
  dict:
    other_subkey: 40
```

The final result will be:

```yaml
other_section:
  key: 3 # Preserved from config1 (non-conflicting)
  foo: bar # Added from config2 (non-conflicting)
  dict:
    subkey: 30 # Preserved from config1 (non-conflicting nested key)
    other_subkey: 40 # Added from config2 (non-conflicting nested key)
```

If both configs had `other_section.foo`, the value from `config2` would win (last entry wins for conflicting keys).

**Note**: The `override` flag changes merge behavior: when specified, the loaded config completely replaces the value at the package path instead of merging. This means all existing keys at that path are removed and replaced with the loaded content.

### Override flag behavior with `_self_`

When `override` is used, the order of entries still matters. If `_self_` is explicitly included in the list, it will be processed in the order specified:

```yaml
mlflow_defaults:
  - load_source_1@other_section: config1# # Merges at other_section
  - override load_source_2@other_section: config2# # Replaces other_section completely
  - _self_ # Merges original config, so original values can override the override
```

In this example:

1. `config1` is merged into `other_section` (preserving existing keys, merging nested dicts)
2. `config2` completely replaces `other_section` (removing all keys from step 1)
3. `_self_` merges the original config, so any keys in the original `other_section` will override/replace what was loaded from `config2`

**Important**: `_self_` is always auto-appended if not explicitly included, regardless of override usage. If you want to override everything from `_self_`, explicitly include `_self_` first, then use `override` at root:

```yaml
mlflow_defaults:
  - _self_ # Explicitly include first
  - override load_source # Then override at root, replacing everything
```

## Implementation:

The proposed implementation is as a stand alone function that dynamically resolves the config:

```python
from mlflow.client import MlflowClient
from omegaconf import DictConfig, OmegaConf

from simplexity.utils.config_utils import dynamic_resolve

def load_mlflow_defaults(cfg: DictConfig) -> DictConfig
    # 1. Parse 'mlflow_defaults' list.
    #    - If '_self_' is missing, append it to the end.

    # 2. Iterate through items:
    #    - If item is '_self_':
    #         Merge original cfg (passed in argument) into the accumulator.
    #    - Else (MLFLOW_ENTRY):
    #         Parse TARGET, PACKAGE, OPTION.
    #         If OPTION is "null":
    #             If "optional" flag is set: continue (ignore)
    #             Else: throw Error ("Config is mandatory but OPTION is null")
    #
    #         Resolve ARTIFACT_PATH and SELECT_PATH from OPTION.
    #
    #         Resolve 'tracking_uri', 'run_id' from cfg[TARGET].
    #         Instantiate MlflowClient.
    #
    #         Download artifact (with caching/tempfile).
    #         Load artifact as DictConfig.
    #
    #         Select subconfig if SELECT_PATH is set.
    #
    #         Merge into accumulator at PACKAGE.

    # 3. Return accumulator.
    ...
```

This could then be included in the `managed_run` decorator

```python
...
cfg = get_config(args, kwargs)
cfg = load_mlflow_defaults(cfg) # <- load subconfigs here
validate_base_config(cfg)
resolve_base_config(cfg, strict=strict)
...
```

Or used as by itself, such as in a notebook:

```python
# %%
cfg = DictConfig({
    "load_source": DictConfig({
        "experiment_id": "9828318895773678"
        "run_id": "93c47bf390aef1273573b9dd53de2d3a"
        "tracking_uri": "databricks"
    }),
    "mlflow_defaults": "load_source",
})

cfg = load_mlflow_defaults(cfg)
```

## Testing

### Test Cases

**Entries**

- Single `ENTRY`
- Multiple `ENTRY` items for different runs
- Multiple `ENTRY` items for the same run
- Multiple `ENTRY` items with shared keys but differing values
  - Resulting value comes from last `ENTRY` in the list with that key
- Explicit `_self_` omitted
  - included implicitly at the end
- Explicit `_self_` before other `ENTRY` item(s)

**Packages**

- Explicit `PACKAGE` omitted
  - Load at config root by default
- Load at specified `PACKAGE` path

**Artifact Paths**

- Explicit `ARTIFACT_PATH` omitted
  - `config.yaml` loaded by default
- Load config specified by `ARTIFACT_PATH`

**Select Paths**

- Explicit `SELECT_PATH` omitted
  - Entire `TARGET` config loaded by default
- Load subconfig specified by `SELECT_PATH`
- Load a single key specified by `SELECT_PATH`

**Options**

- Explicit `optional` omitted
  - Throws exception if `OPTION` is `null`
  - Throws exception if there is an issue loading given `OPTION`
- Explicit `optional` specified
  - Accepts `null` value for `OPTION` effectively skipping that `ENTRY`
  - Loads valid `OPTION` normally
  - Skips that `ENTRY` if there is an issue loading given `OPTION`
- Explicit `override` omitted
  - No effect
- Explicit `override` specified
  - No effect

### Unit Tests

- There should be individual test functions for each of the [Test Cases]
- Each case should be tested in relative isolation to the extent that that is feasible (use simplest defaults for components not under test)
- Calls to `MlflowClient.download_artifacts` should be mocked with the side effect of saving a yaml file with content needed by the test in a temp directory to avoid this dependency not under test

### Integration Tests

- Should aim to broad coverage of the possibilites in [Test Cases]
- Should test realistic workflows (such as corresponding to the structures of [Examples]) with individual tests typically covering multiple non-default elements of [Test Cases]
- Should avoid any mocking (use local MLflow with `sqlite://` in a temp directory as the `tracking_uri`)
