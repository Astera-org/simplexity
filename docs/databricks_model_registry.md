# Databricks Model Registry Guide

This project targets the Databricks **Workspace Model Registry** by default because many subscriptions (including ours) do not provide Unity Catalog access. The integration is designed so that switching to Unity Catalog later only requires configuration changes—no code changes.

## Default Behaviour

- Hydra config `logging=mlflow_logger` sets `tracking_uri=databricks` and `registry_uri=databricks`.
- `simplexity.utils.mlflow_utils.resolve_registry_uri` downgrades Unity Catalog URIs (``databricks-uc``) to workspace URIs when `allow_workspace_fallback=True` (the default) and emits a warning so you know a downgrade happened.
- `MLFlowLogger` and `MLFlowPersister.from_experiment` both call `resolve_registry_uri`, so any code path that uses Simplexity helpers gets the same fallback logic.
- `examples/mlflow_workspace_registry_demo.py` mirrors this behaviour and can be used to sanity-check Databricks connectivity.

## Preparing for a Future Unity Catalog Migration

To keep migration friction low we expose an `allow_workspace_fallback` flag everywhere MLflow clients are created.

- **Logger config** (`simplexity/configs/logging/mlflow_logger.yaml`):
  - Set `registry_uri: databricks-uc` once your workspace is UC-enabled.
  - Flip `allow_workspace_fallback: false` to stop the automatic downgrade.
- **Programmatic use**: `MLFlowLogger(..., allow_workspace_fallback=False)` or `MLFlowPersister.from_experiment(..., allow_workspace_fallback=False)` preserves Unity Catalog URIs.
- **Environment variables**: you can still rely on `MLFLOW_TRACKING_URI` / `MLFLOW_REGISTRY_URI`. When fallback is disabled those values are forwarded unchanged.

Because the flag defaults to `True`, current jobs continue working even if a Unity Catalog URI is supplied accidentally—Simplexity automatically falls back to the workspace registry and logs a warning. When you are ready to migrate, toggling the flag allows UC usage without touching the codebase.

## Suggested Migration Checklist

1. **Enable Unity Catalog in Databricks** and make sure the MLflow registry permissions are set up (see the official Databricks migration guide).
2. **Create the Unity Catalog equivalents** of any workspace-registered models if you plan to keep history—Databricks provides automated migration jobs for this.
3. **Update configuration**:
   - Set `registry_uri` (and optionally `tracking_uri`) to the appropriate `databricks-uc` endpoint.
   - Set `allow_workspace_fallback: false` to surface real UC connectivity errors instead of silently downgrading.
4. **Smoke test** using `examples/mlflow_workspace_registry_demo.py` with the updated config. The script will now run against UC and should register the demo model there.
5. **Monitor warnings**: once fallback is disabled, any remaining downgrade warnings indicate stale configs or code paths that still pass the workspace URI.

## Operational Notes

- Keeping fallback enabled during the transition phase is helpful because it avoids runtime failures, but remember that models will continue to land in the workspace registry until you turn it off.
- After migration you can remove the fallback flag entirely or leave it `False` so that future regressions are caught early.
- If you need parallel workspace/UC logging (for validation) you can run two Hydrated jobs with different logger configs—no application code changes are required.
