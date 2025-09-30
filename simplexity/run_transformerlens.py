from contextlib import nullcontext

import hydra
from omegaconf import DictConfig
import jax
import torch
import torch.nn.functional as F

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.utils.hydra import typed_instantiate


def _train_step(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Ensure dtype and device are correct
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    inputs = inputs.to(device)
    labels = labels.to(device)
    if inputs.dtype != torch.long:
        inputs = inputs.long()
    logits = model(inputs)
    vocab_size = logits.shape[2]
    logits_reshaped = logits.view(-1, vocab_size)
    labels_reshaped = labels.view(-1).long()
    loss_tensor = F.cross_entropy(logits_reshaped, labels_reshaped)
    return loss_tensor, float(loss_tensor.item())


@hydra.main(config_path="configs", config_name="transformerlens_mess3.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a TransformerLens model on mess3 data."""
    assert isinstance(cfg, DictConfig)
    validate_config(cfg)

    # Logger
    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg, resolve=True)
        logger.log_params(cfg)
    else:
        logger = None

    # Data generators
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    # Model (PyTorch / TransformerLens)
    model = typed_instantiate(cfg.predictive_model.instance, torch.nn.Module)

    # Optimizer
    optimizer = typed_instantiate(cfg.training.optimizer.instance, torch.optim.Optimizer, params=model.parameters())

    # Training
    key = jax.random.PRNGKey(cfg.training.seed)
    gen_state = training_data_generator.initial_state
    # Note: states are in JAX; batch states are returned from generator
    loss_value = 0.0

    for step in range(1, cfg.training.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_state_batch = None
        if step == 1:
            # Initialize batch of generator states on first iteration
            import jax.numpy as jnp

            gen_state_batch = jnp.repeat(gen_state[None, :], cfg.training.batch_size, axis=0)
        else:
            # Reuse previous batched states
            gen_state_batch = gen_states

        gen_states, inputs, labels = generate_data_batch(
            gen_state_batch,
            training_data_generator,
            cfg.training.batch_size,
            cfg.training.sequence_len,
            gen_key,
            bos_token=cfg.training_data_generator.bos_token,
            eos_token=cfg.training_data_generator.eos_token,
        )

        model.train()
        optimizer.zero_grad()

        loss_tensor, loss_value = _train_step(model, inputs, labels)
        loss_tensor.backward()
        optimizer.step()

        if logger and cfg.training.log_every and step % cfg.training.log_every == 0:
            logger.log_metrics(step, {"loss": loss_value})

        # Validation
        if (
            logger
            and validation_data_generator is not None
            and cfg.training.validate_every
            and step % cfg.training.validate_every == 0
            and cfg.validation is not None
        ):
            import jax.numpy as jnp

            model.eval()
            with torch.no_grad():
                val_key = key
                val_gen_state = validation_data_generator.initial_state
                val_gen_states = jnp.repeat(val_gen_state[None, :], cfg.validation.batch_size, axis=0)
                val_losses = []
                for _ in range(cfg.validation.num_steps):
                    val_key, val_gen_key = jax.random.split(val_key)
                    val_gen_states, val_inputs, val_labels = generate_data_batch(
                        val_gen_states,
                        validation_data_generator,
                        cfg.validation.batch_size,
                        cfg.validation.sequence_len,
                        val_gen_key,
                        bos_token=validation_bos_token,
                        eos_token=validation_eos_token,
                    )
                    val_loss_tensor, val_loss = _train_step(model, val_inputs, val_labels)
                    val_losses.append(val_loss)
                avg_val_loss = sum(val_losses) / len(val_losses)
                logger.log_metrics(step, {"validation/loss": avg_val_loss})

    if logger:
        logger.close()

    return loss_value


if __name__ == "__main__":
    train_model()

