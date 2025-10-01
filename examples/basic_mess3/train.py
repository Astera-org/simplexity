"""Training script for TransformerLens on mess3 data.

This example demonstrates how to use Simplexity to train a TransformerLens model
on synthetic data from a Hidden Markov Model (mess3). All artifacts including
model checkpoints are uploaded to MLflow.
"""

import tempfile
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.utils.config_resolution import compute_generator_sequence_length, compute_model_vocab_size
from simplexity.utils.hydra import typed_instantiate
from simplexity.utils.pytorch_utils import resolve_device


def train_with_device_support(
    model: torch.nn.Module,
    training_cfg,
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    bos_token: int | None = None,
) -> tuple[torch.nn.Module, float]:
    """Train a PyTorch model with MLflow artifact upload."""
    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    key = jax.random.PRNGKey(training_cfg.seed)

    optimizer = typed_instantiate(training_cfg.optimizer.instance, torch.optim.Optimizer, params=model.parameters())

    gen_state = training_data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], training_cfg.batch_size, axis=0)
    loss_value = 0.0

    start_time = time.time()
    tokens_processed = 0

    for step in range(1, training_cfg.num_steps + 1):
        step_start_time = time.time()
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            training_data_generator,
            training_cfg.batch_size,
            training_cfg.sequence_len,
            gen_key,
            bos_token=bos_token,
            eos_token=None,
        )

        inputs = inputs.to(device)
        labels = labels.to(device)

        model.train()
        optimizer.zero_grad()

        logits = model(inputs)

        vocab_size = logits.shape[2]
        logits_reshaped = logits.view(-1, vocab_size)
        labels_reshaped = labels.view(-1).long()

        loss_tensor = F.cross_entropy(logits_reshaped, labels_reshaped)
        loss = loss_tensor.item()
        loss_value = loss

        loss_tensor.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm**0.5

        optimizer.step()

        tokens_processed += training_cfg.batch_size * training_cfg.sequence_len

        step_time = time.time() - step_start_time
        tokens_per_second = (training_cfg.batch_size * training_cfg.sequence_len) / step_time if step_time > 0 else 0

        current_lr = optimizer.param_groups[0]["lr"]

        memory_stats = {}
        if device.type == "cuda":
            memory_stats = {
                "memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            }

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "tokens_per_second": tokens_per_second,
            "step_time_seconds": step_time,
            **memory_stats,
        }

        if logger and training_cfg.log_every and step % training_cfg.log_every == 0:
            elapsed_time = time.time() - start_time
            metrics["total_tokens"] = tokens_processed
            metrics["total_time_seconds"] = elapsed_time
            metrics["avg_tokens_per_second"] = tokens_processed / elapsed_time if elapsed_time > 0 else 0

            logger.log_metrics(step, metrics)
            print(
                f"Step {step}: loss = {loss:.4f}, grad_norm = {grad_norm:.2f}, "
                f"lr = {current_lr:.6f}, tokens/s = {tokens_per_second:.1f}"
            )

        # NOTE: In future, extract this to simplexity.persistence.mlflow_persister.MLflowPersister
        if (
            logger
            and training_cfg.checkpoint_every
            and step % training_cfg.checkpoint_every == 0
            and hasattr(logger, "_client")
            and hasattr(logger, "_run_id")
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = Path(temp_dir) / f"step_{step}" / "model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

                logger._client.log_artifact(logger._run_id, str(checkpoint_path), artifact_path="checkpoints")  # type: ignore[attr-defined]
            print(f"Checkpoint uploaded to MLflow at step {step}")

    return model, loss_value


@hydra.main(config_path="configs", config_name="transformerlens_mess3.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a TransformerLens model on mess3 data with MLflow artifact storage."""
    assert isinstance(cfg, DictConfig)

    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)

    from transformer_lens import HookedTransformer, HookedTransformerConfig

    config_dict = dict(cfg.predictive_model.instance.cfg)  # type: ignore[attr-defined]
    config_dict.pop("_target_", None)

    device_str = config_dict.get("device", "auto")
    config_dict["device"] = resolve_device(device_str)

    use_bos = cfg.training_data_generator.use_bos  # type: ignore[attr-defined]
    bos_token = training_data_generator.vocab_size if use_bos else None

    model_vocab_size = compute_model_vocab_size(training_data_generator.vocab_size, use_bos, use_eos=False)
    config_dict["d_vocab"] = model_vocab_size

    model_n_ctx = config_dict["n_ctx"]
    training_sequence_len = compute_generator_sequence_length(model_n_ctx, use_bos)

    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.training.sequence_len = training_sequence_len
    cfg.training_data_generator.vocab_size = model_vocab_size
    cfg.training_data_generator.bos_token = bos_token
    OmegaConf.set_struct(cfg, True)

    validate_config(cfg)

    print(f"Resolved device: {config_dict['device']}")
    print(
        f"Computed model d_vocab: {model_vocab_size} "
        f"(from generator vocab_size={training_data_generator.vocab_size}, use_bos={use_bos})"
    )
    print(
        f"Computed training sequence_len: {training_sequence_len} (from model n_ctx={model_n_ctx}, use_bos={use_bos})"
    )
    print(f"Computed bos_token: {bos_token}")

    model_config = HookedTransformerConfig(**config_dict)
    model = HookedTransformer(model_config)

    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg, resolve=True)
        logger.log_params(cfg)
        logger.log_params(
            {
                "computed/model_vocab_size": model_vocab_size,
                "computed/training_sequence_len": training_sequence_len,
                "computed/bos_token": bos_token if bos_token is not None else "null",
                "computed/use_bos": use_bos,
            }
        )
        logger.log_git_info()

        if hasattr(logger, "_client") and hasattr(logger, "_run_id"):
            repo_root = Path.cwd()
            artifacts = []
            if (uv_lock := repo_root / "uv.lock").exists():
                artifacts.append((uv_lock, "deps/uv.lock"))
            if (pyproject := repo_root / "pyproject.toml").exists():
                artifacts.append((pyproject, "deps/pyproject.toml"))

            for src, dest in artifacts:
                artifact_path = str(Path(dest).parent)
                logger._client.log_artifact(logger._run_id, str(src), artifact_path=artifact_path)  # type: ignore[attr-defined]
    else:
        logger = None

    _, loss = train_with_device_support(
        model,
        cfg.training,
        training_data_generator,
        logger,
        bos_token=bos_token,
    )

    if logger:
        logger.close()

    return loss


if __name__ == "__main__":
    train_model()
