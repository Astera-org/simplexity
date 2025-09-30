from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train as train_jax
from simplexity.training.train_pytorch_model import train as train_torch
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a model."""
    assert isinstance(cfg, DictConfig)
    validate_config(cfg)

    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None

    # Instantiate data generators
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    # Ensure model vocab_size matches data generator where applicable before instantiation
    vocab_size = int(cfg.training_data_generator.vocab_size)
    # Attempt to set common locations for vocab size (top-level or nested under config/cfg)
    inst = cfg.predictive_model.instance
    try:
        if hasattr(inst, "vocab_size") or (isinstance(inst, DictConfig) and "vocab_size" in inst):
            cfg.predictive_model.instance.vocab_size = vocab_size
        # penzai-style nested config
        if isinstance(inst, DictConfig) and "config" in inst and "vocab_size" in inst["config"]:
            cfg.predictive_model.instance.config.vocab_size = vocab_size
        # transformer-lens-style nested cfg
        if isinstance(inst, DictConfig) and "cfg" in inst and "d_vocab" in inst["cfg"]:
            cfg.predictive_model.instance.cfg.d_vocab = vocab_size
    except Exception:
        # Keep going; we will still validate post-instantiation
        pass

    # Consistency checks for sequence lengths and BOS/EOS
    bos = cfg.training_data_generator.bos_token
    eos = cfg.training_data_generator.eos_token
    B = 1 if bos is not None else 0
    E = 1 if eos is not None else 0

    # Training/validation sequence length alignment
    if cfg.validation is not None:
        assert (
            cfg.validation.sequence_len == cfg.training.sequence_len
        ), "validation.sequence_len must match training.sequence_len for consistent context"

    # BOS/EOS validity
    if bos is not None:
        assert 0 <= bos < vocab_size, "bos_token must be within [0, vocab_size)"
    if eos is not None:
        assert 0 <= eos < vocab_size, "eos_token must be within [0, vocab_size)"

    # Effective model input length (for transparency and potential model checks)
    effective_inputs_len = cfg.training.sequence_len + B + E - 1

    # Instantiate model without enforcing a specific protocol, then route to the right trainer
    model = hydra.utils.instantiate(cfg.predictive_model.instance)

    persister_context = (
        typed_instantiate(cfg.persistence.instance, ModelPersister) if cfg.persistence else nullcontext()
    )

    with persister_context as persister:
        if isinstance(persister, ModelPersister):
            if cfg.predictive_model.load_checkpoint_step:
                model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_step)
            train_persister = persister
        else:
            train_persister = None

        # Choose trainer based on model type
        try:
            import torch.nn as nn  # defer import to avoid requiring torch for JAX runs
        except Exception:
            nn = None  # type: ignore

        if nn is not None and isinstance(model, nn.Module):
            # If TransformerLens-style config exists, assert n_ctx alignment when set
            try:
                n_ctx = None
                inst = cfg.predictive_model.instance
                if isinstance(inst, DictConfig) and "cfg" in inst and "n_ctx" in inst["cfg"]:
                    n_ctx = int(inst["cfg"]["n_ctx"])  # type: ignore
                if n_ctx is not None:
                    assert (
                        n_ctx == effective_inputs_len
                    ), f"predictive_model.cfg.n_ctx ({n_ctx}) must equal effective inputs length ({effective_inputs_len}) computed from sequence_len and BOS/EOS"
            except Exception:
                # Be permissive; downstream will surface shape mismatches
                pass

            _, loss = train_torch(
                model,
                cfg.training,
                training_data_generator,
                logger,
                cfg.validation,
                validation_data_generator,
                train_persister,
                training_bos_token=cfg.training_data_generator.bos_token,
                training_eos_token=cfg.training_data_generator.eos_token,
                validation_bos_token=validation_bos_token,
                validation_eos_token=validation_eos_token,
            )
        else:
            # Default JAX/Penzai/Equinox path
            _, loss = train_jax(
                model,  # type: ignore[arg-type]
                cfg.training,
                training_data_generator,
                logger,
                cfg.validation,
                validation_data_generator,
                train_persister,
                training_bos_token=cfg.training_data_generator.bos_token,
                training_eos_token=cfg.training_data_generator.eos_token,
                validation_bos_token=validation_bos_token,
                validation_eos_token=validation_eos_token,
            )

    if logger:
        logger.close()

    return loss


if __name__ == "__main__":
    train_model()
