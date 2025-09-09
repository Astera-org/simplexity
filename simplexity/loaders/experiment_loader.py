from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig

from simplexity.logging.mlflow_reader import MLflowRunReader
from simplexity.logging.run_reader import RunReader
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.hydra import typed_instantiate


@dataclass
class ExperimentLoader:
    """High-level loader for reconstructing models and reading run data."""

    reader: RunReader
    config_path: str | None = None
    _cached_config: DictConfig | None = None

    # --- Constructors ---
    @classmethod
    def from_mlflow(cls, run_id: str, tracking_uri: str | None = None, config_path: str | None = None) -> "ExperimentLoader":
        reader = MLflowRunReader(run_id=run_id, tracking_uri=tracking_uri)
        return cls(reader=reader, config_path=config_path)

    # --- Accessors ---
    def load_config(self) -> DictConfig:
        if self._cached_config is None:
            self._cached_config = self.reader.get_config()
        return self._cached_config

    def load_metrics(self, pattern: str | None = None):
        return self.reader.get_metrics(pattern=pattern)

    # --- Helper methods ---
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual PyTorch device."""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    # --- Model reconstruction ---
    def _instantiate_model_and_persister(self) -> tuple[PredictiveModel, ModelPersister | None, DictConfig]:
        cfg = self.load_config()
        try:
            # Handle device resolution for 'auto' setting
            model_config = dict(cfg.predictive_model.instance)
            if 'cfg' in model_config and hasattr(model_config['cfg'], 'device'):
                if model_config['cfg']['device'] == 'auto':
                    model_config = dict(model_config)
                    model_config['cfg'] = dict(model_config['cfg'])
                    model_config['cfg']['device'] = self._resolve_device('auto')
            
            model = typed_instantiate(model_config, PredictiveModel)
        except Exception as e:
            raise RuntimeError(
                "Failed to instantiate predictive model from run config.\n"
                "Ensure the model's Python package is installed (e.g., `transformer_lens`).\n"
                f"Underlying error: {e}"
            ) from e
        persister: ModelPersister | None
        if cfg.persistence:
            try:
                # Override config_filename if custom config_path is provided
                persister_config = dict(cfg.persistence.instance)
                if self.config_path and 'config_filename' in persister_config:
                    persister_config['config_filename'] = self.config_path
                persister = typed_instantiate(persister_config, ModelPersister)
            except Exception as e:
                raise RuntimeError(
                    "Failed to instantiate persister from run config.\n"
                    "If using S3, ensure credentials/config are available (e.g., config.ini or env).\n"
                    f"Underlying error: {e}"
                ) from e
        else:
            persister = None
        return model, persister, cfg

    def _instantiate_persister_only(self) -> ModelPersister | None:
        cfg = self.load_config()
        if not cfg.persistence:
            return None
        try:
            # Override config_filename if custom config_path is provided
            persister_config = dict(cfg.persistence.instance)
            if self.config_path and 'config_filename' in persister_config:
                persister_config['config_filename'] = self.config_path
            return typed_instantiate(persister_config, ModelPersister)
        except Exception:
            # Best-effort: return None if we cannot construct the persister (missing creds, etc.)
            return None

    def list_checkpoints(self) -> list[int]:
        persister = self._instantiate_persister_only()
        return persister.list_checkpoints() if persister else []

    def latest_checkpoint(self) -> int | None:
        persister = self._instantiate_persister_only()
        return persister.latest_checkpoint() if persister else None

    def load_model(self, step: int | Literal["latest"] = "latest") -> PredictiveModel:
        model, persister, _ = self._instantiate_model_and_persister()
        if not persister:
            raise RuntimeError("No persistence configuration found in run config; cannot load checkpoints.")

        target_step: int
        if step == "latest":
            latest = persister.latest_checkpoint()
            if latest is None:
                raise RuntimeError("No checkpoints found for this run.")
            target_step = latest
        else:
            if not persister.checkpoint_exists(step):
                raise RuntimeError(f"Requested checkpoint step {step} does not exist.")
            target_step = step

        return persister.load_weights(model, target_step)
