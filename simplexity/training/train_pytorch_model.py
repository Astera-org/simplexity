import jax
import jax.numpy as jnp

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.evaluation.evaluate_pytorch_model import evaluate
from simplexity.generative_processes.generative_process import GenerativeProtocol
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.utils.hydra import typed_instantiate
from simplexity.utils.pytorch_utils import torch_to_jax

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


def train(
    model: torch.nn.Module,
    training_cfg: TrainConfig,
    training_data_generator: GenerativeProtocol,
    logger: Logger | None = None,
    validation_cfg: ValidateConfig | None = None,
    validation_data_generator: GenerativeProtocol | None = None,
    persister: ModelPersister | None = None,
    training_bos_token: int | None = None,
    training_eos_token: int | None = None,
    validation_bos_token: int | None = None,
    validation_eos_token: int | None = None,
) -> tuple[torch.nn.Module, float]:
    """Train a PyTorch model on a generative process."""
    key = jax.random.PRNGKey(training_cfg.seed)

    optimizer = typed_instantiate(training_cfg.optimizer.instance, torch.optim.Optimizer, params=model.parameters())

    gen_state = training_data_generator.initial_state
    if isinstance(gen_state, jax.Array):
        gen_states = jnp.repeat(gen_state[None, :], training_cfg.batch_size, axis=0)
    else:
        gen_states = gen_state
    loss_value = 0.0

    for step in range(1, training_cfg.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            training_data_generator,
            training_cfg.batch_size,
            training_cfg.sequence_len,
            gen_key,
            bos_token=training_bos_token,
            eos_token=training_eos_token,
        )

        model.train()
        optimizer.zero_grad()

        logits = model(inputs)

        vocab_size = logits.shape[2]
        logits_reshaped = logits.view(-1, vocab_size)
        labels_reshaped = labels.view(-1).long()  # Ensure labels are long type for cross entropy

        loss_tensor = F.cross_entropy(logits_reshaped, labels_reshaped)
        loss = torch_to_jax(loss_tensor).item()
        loss_value = loss

        loss_tensor.backward()
        optimizer.step()

        metrics = {"loss": loss}

        if logger:
            if training_cfg.log_every and step % training_cfg.log_every == 0:
                logger.log_metrics(step, metrics)
            if (
                validation_cfg
                and validation_data_generator
                and training_cfg.validate_every
                and step % training_cfg.validate_every == 0
            ):
                validation_metrics = evaluate(
                    model=model,
                    cfg=validation_cfg,
                    data_generator=validation_data_generator,
                    bos_token=validation_bos_token,
                    eos_token=validation_eos_token,
                )
                validation_metrics = {f"validation/{k}": v for k, v in validation_metrics.items()}
                logger.log_metrics(step, validation_metrics)
        if persister and training_cfg.checkpoint_every and step % training_cfg.checkpoint_every == 0:
            persister.save_weights(model, step)

    return model, loss_value
