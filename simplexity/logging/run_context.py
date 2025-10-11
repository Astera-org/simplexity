"""RunContext: standardized MLflow run provenance and setup.

This context manager centralizes common experiment-setup patterns:
- Instantiate a `Logger` and `MLFlowPersister` from Hydra config
- Log full provenance (resolved config, pre‑mutation snapshot, Hydra artifacts)
- Log hierarchical params, git info, environment, and optional source script
- Provide `ctx.logger` and `ctx.persister` for training/eval code

Typical usage (inside a Hydra @main):

    from omegaconf import OmegaConf
    from simplexity.logging.run_context import RunContext

    pre_cfg_unresolved = OmegaConf.to_container(cfg, resolve=False)
    with RunContext(
        cfg,
        pre_cfg_unresolved=pre_cfg_unresolved,
        source_relpath="mess3_simple/run.py",
        tags={"run.kind": "train", "task": "mess3_simple"},
    ) as ctx:
        logger = ctx.logger
        persister = ctx.persister
        # training loop ...

Notes:
- All provenance logging toggles default to True; you can disable for tests.
- `tags` are applied if provided; use `apply_standard_tags` helper to add a baseline set.
- Environment artifacts logging occurs only if the logger implements it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from simplexity.logging.logger import Logger
from simplexity.persistence.mlflow_persister import MLFlowPersister
from simplexity.predictive_models.types import ModelFramework
from simplexity.utils.hydra import typed_instantiate


log = logging.getLogger(__name__)


class RunContext:
    """Context manager for standardized run provenance and resources.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing `logging.instance` and `persistence.instance`.
    pre_cfg_unresolved : Mapping[str, Any] | None, optional
        Snapshot of the config before runtime mutations (for debugging interpolations), by default None.
    source_relpath : str | None, optional
        Source script path relative to repo root (e.g., "mess3_simple/run.py"), by default None.
    model_framework : ModelFramework, optional
        Model framework enum for persistence routing, by default ModelFramework.Equinox.
    log_hydra_artifacts : bool, optional
        Whether to upload `.hydra/` artifacts, by default True.
    log_git_info : bool, optional
        Whether to log git tags (commit/branch/dirty), by default True.
    log_environment : bool, optional
        Whether to log environment artifacts, by default True.
    tags : Mapping[str, Any] | None, optional
        Optional tags to apply to the run, by default None.
    logger : Logger | None, optional
        Optional pre-instantiated logger; if None, created from cfg, by default None.
    persister : MLFlowPersister | None, optional
        Optional pre-instantiated persister; if None, created from cfg, by default None.
    strict : bool, optional
        If True, raise on provenance failures (e.g., missing source file). If False, warn, by default False.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        pre_cfg_unresolved: Mapping[str, Any] | None = None,
        source_relpath: str | None = None,
        model_framework: ModelFramework = ModelFramework.Equinox,
        log_hydra_artifacts: bool = True,
        log_git_info: bool = True,
        log_environment: bool = True,
        tags: Mapping[str, Any] | None = None,
        logger: Logger | None = None,
        persister: MLFlowPersister | None = None,
        strict: bool = False,
    ) -> None:
        self.cfg = cfg
        self.pre_cfg_unresolved = pre_cfg_unresolved
        self.source_relpath = source_relpath
        self.model_framework = model_framework
        self.log_hydra_artifacts_flag = log_hydra_artifacts
        self.log_git_info_flag = log_git_info
        self.log_environment_flag = log_environment
        self.tags = dict(tags) if tags else None
        self.strict = strict

        self.logger: Logger | None = logger
        self.persister: MLFlowPersister | None = persister

    def __enter__(self) -> "RunContext":
        # 1) Instantiate logger/persister if not provided
        if self.logger is None:
            self.logger = typed_instantiate(self.cfg.logging.instance, Logger)
        if self.persister is None:
            # Instantiate persister only if a persistence config is present
            try:
                has_persistence = hasattr(self.cfg, "persistence") and hasattr(self.cfg.persistence, "instance")  # type: ignore[attr-defined]
            except Exception:
                has_persistence = False

            if has_persistence:
                # Prefer using from_logger when available in cfg (common pattern)
                try:
                    self.persister = typed_instantiate(
                        self.cfg.persistence.instance,
                        MLFlowPersister,
                        logger=self.logger,
                        model_framework=self.model_framework,
                    )
                except Exception:
                    # Fallback: instantiate with explicit client+run_id if config expects that
                    self.persister = typed_instantiate(
                        self.cfg.persistence.instance,
                        MLFlowPersister,
                        model_framework=self.model_framework,
                    )

        # 2) Log resolved config and optional pre‑mutation snapshot
        self.logger.log_config(self.cfg, resolve=True)
        if self.pre_cfg_unresolved is not None:
            try:
                self.logger.log_json_artifact(
                    dict(self.pre_cfg_unresolved),
                    "config_pre_update_unresolved.json",
                )
            except Exception as e:
                if self.strict:
                    raise
                log.warning("Failed to log pre-mutation config snapshot: %s", e)

        # 3) Hydra artifacts (.hydra/)
        if self.log_hydra_artifacts_flag:
            self._log_hydra_artifacts()

        # 4) Hierarchical params (flattened by logger implementation)
        try:
            self.logger.log_params(self.cfg)
        except Exception as e:
            if self.strict:
                raise
            log.warning("Failed to log params: %s", e)

        # 5) Git info
        if self.log_git_info_flag:
            try:
                self.logger.log_git_info()
            except Exception as e:
                if self.strict:
                    raise
                log.warning("Failed to log git info: %s", e)

        # 6) Environment artifacts (if supported by logger)
        if self.log_environment_flag and hasattr(self.logger, "log_environment_artifacts"):
            try:
                # type: ignore[attr-defined]
                self.logger.log_environment_artifacts()
            except Exception as e:
                if self.strict:
                    raise
                log.warning("Failed to log environment artifacts: %s", e)

        # 7) Source script
        if self.source_relpath:
            self._log_source_script(self.source_relpath)

        # 8) Tags
        if self.tags:
            try:
                self.logger.log_tags(self.tags)
            except Exception as e:
                if self.strict:
                    raise
                log.warning("Failed to log tags: %s", e)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Best-effort cleanup on persister if available
        cleanup = getattr(self.persister, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as e:
                log.warning("Persister cleanup failed: %s", e)
        # Do not suppress exceptions
        return False

    # Public helper
    def apply_standard_tags(self, kind: str, task: str, extras: Mapping[str, Any] | None = None) -> None:
        """Apply a baseline set of tags (non-fatal on failures)."""
        payload: dict[str, Any] = {"run.kind": kind, "task": task}
        if extras:
            payload.update(extras)
        try:
            self.logger.log_tags(payload)  # type: ignore[union-attr]
        except Exception as e:
            if self.strict:
                raise
            log.warning("Failed to apply standard tags: %s", e)

    # Internal helpers
    def _log_hydra_artifacts(self) -> None:
        try:
            hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
        except Exception:
            # Outside Hydra context; nothing to do
            return
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            p = hydra_dir / name
            if p.exists():
                try:
                    self.logger.log_artifact(str(p), artifact_path=".hydra")  # type: ignore[union-attr]
                except Exception as e:
                    if self.strict:
                        raise
                    log.warning("Failed to log Hydra artifact %s: %s", p, e)

    def _log_source_script(self, relpath: str) -> None:
        repo_root = Path(get_original_cwd())
        script_path = repo_root / relpath
        if not script_path.exists():
            msg = f"Source script not found: {script_path}"
            if self.strict:
                raise FileNotFoundError(msg)
            log.warning(msg)
            return
        try:
            self.logger.log_artifact(str(script_path), artifact_path="source")  # type: ignore[union-attr]
        except Exception as e:
            if self.strict:
                raise
            log.warning("Failed to log source script: %s", e)
