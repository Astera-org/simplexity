"""Demo script that trains a tiny model and surfaces activation visualizations."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import torch
import tqdm.auto as tqdm
import yaml
from tempfile import TemporaryDirectory
from torch.nn import Module as PytorchModel
from torch.nn import functional as F

import simplexity
from examples.configs.demo_config_with_visuals import Config
from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.generative_processes.torch_generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)

DEMO_DIR = Path(__file__).parent


def _configure_logging() -> None:
    logging_config_path = DEMO_DIR / "configs" / "logging.yaml"
    if logging_config_path.exists():
        with logging_config_path.open(encoding="utf-8") as source:
            logging_cfg = yaml.safe_load(source)
        import logging.config

        logging.config.dictConfig(logging_cfg)
    else:
        import logging

        logging.basicConfig(level=logging.INFO)


@hydra.main(config_path=str(DEMO_DIR / "configs"), config_name="demo_config_with_visuals.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=True)
def main(cfg: Config, components: simplexity.Components) -> None:
    tracker = components.get_activation_tracker()
    logger = components.get_logger()
    generative_process = components.get_generative_process()
    initial_state = components.get_initial_state()
    model = components.get_predictive_model()
    optimizer = components.get_optimizer()

    if tracker is None or not isinstance(tracker, ActivationTracker):
        raise RuntimeError("Activation tracker component is required for this demo.")
    if generative_process is None or initial_state is None:
        raise RuntimeError("Generative process and initial state must be configured.")
    if model is None or not isinstance(model, PytorchModel):
        raise RuntimeError("Demo requires a PyTorch predictive model.")
    if optimizer is None:
        raise RuntimeError("Optimizer configuration is required.")
    if not hasattr(model, "run_with_cache"):
        raise RuntimeError("Predictive model must expose a run_with_cache method.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = model.to(device)

    batch_size = cfg.generative_process.batch_size or 16
    sequence_len = cfg.generative_process.sequence_len or 8
    training_steps = max(1, cfg.training_steps)
    validate_every = max(1, cfg.validate_every)
    if logger is not None:
        # Use a temporary directory for visualization logging
        visualization_root = Path(TemporaryDirectory().name)
    else:
        if cfg.visualization_dir is None:
            raise ValueError("Visualizations requested but no logger or visualization_dir configured.")
        visualization_root = Path(cfg.visualization_dir)
        visualization_root.mkdir(parents=True, exist_ok=True)

    batched_state = _prepare_initial_states(initial_state, batch_size)
    progress = tqdm.tqdm(total=training_steps, desc="training", unit="step")
    val_loss = float("nan")

    for step in range(0, training_steps + 1):

        if step % validate_every == 0:
            val_loss = _run_validation(
                tracker,
                generative_process,
                model,
                logger,
                batched_state,
                cfg,
                metadata_step=step,
                device=device,
                visualization_root=visualization_root,
            )
        
        key = jax.random.key(cfg.seed + step)
        batched_state, inputs_torch, labels = generate_data_batch(
            batched_state,
            generative_process,
            batch_size,
            sequence_len,
            key,
            bos_token=cfg.generative_process.bos_token,
            eos_token=cfg.generative_process.eos_token,
        )

        logits = model(inputs_torch.to(device))
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1).long().to(device),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if logger and hasattr(logger, "log_metrics"):
            logger.log_metrics(step=step, metric_dict={"train/loss": float(loss.detach().cpu().item())})

        progress.set_postfix({"train_loss": f"{float(loss.detach().cpu().item()):.4f}", "val_loss": f"{val_loss:.4f}"})
        progress.update(1)


def _prepare_initial_states(initial_state, batch_size: int) -> jax.Array:
    state = jnp.asarray(initial_state)
    if state.ndim == 1:
        return jnp.repeat(state[None, :], batch_size, axis=0)
    if state.shape[0] != batch_size:
        raise ValueError("Initial state batch dimension does not match configured batch_size.")
    return state


def _run_validation(
    tracker: ActivationTracker,
    generative_process,
    model,
    logger,
    initial_state,
    cfg: Config,
    *,
    metadata_step: int,
    device,
    visualization_root: Path,
) -> float:
    batch_size = cfg.generative_process.batch_size or 1
    sequence_len = cfg.generative_process.sequence_len or 1
    batched_state = _prepare_initial_states(initial_state, batch_size)
    key = jax.random.key(cfg.seed + metadata_step)

    with torch.no_grad():
        _, belief_states, prefix_probs, inputs_torch, labels = generate_data_batch_with_full_history(
            batched_state,
            generative_process,
            batch_size,
            sequence_len,
            key,
            bos_token=cfg.generative_process.bos_token,
            eos_token=cfg.generative_process.eos_token,
        )
        tokens_len = inputs_torch.shape[1]
        tokens = jnp.array(inputs_torch.cpu().numpy())
        belief_states = jnp.array(belief_states)[:, :tokens_len]
        prefix_probs = jnp.array(prefix_probs)[:, :tokens_len]

        logits, cache = model.run_with_cache(inputs_torch.to(device))
        activations = _collect_layer_activations(cache, model)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1).long().to(device),
        )
        val_loss = float(loss.detach().cpu().item())

    scalars, _, visualizations = tracker.analyze(
        inputs=tokens,
        beliefs=belief_states,
        probs=prefix_probs,
        activations=activations,
        step=metadata_step,
    )

    _log_validation_metrics(logger, scalars, val_loss, metadata_step)
    figs_to_paths = tracker.save_visualizations(visualizations, visualization_root, metadata_step)
    # Optionally log visualizations to the logger
    if logger is not None:
        for plot_name, path in figs_to_paths.items():
            if hasattr(logger, "log_artifact"):
                logger.log_artifact(str(path), artifact_path=f"activation_plots/{plot_name.split('/')[0]}")
    return val_loss


def _collect_layer_activations(cache, model) -> dict[str, jax.Array]:
    activations: dict[str, jax.Array] = {k: v for k, v in cache.items() if "resid" in k}
    return activations


def _log_validation_metrics(logger, scalars: dict[str, float], val_loss: float, step: int) -> None:
    if logger is None or not hasattr(logger, "log_metrics"):
        return
    metric_dict = {f"val/{key}": value for key, value in scalars.items()}
    metric_dict["val/loss"] = val_loss
    logger.log_metrics(step=step, metric_dict=metric_dict)
if __name__ == "__main__":
    _configure_logging()
    main()
