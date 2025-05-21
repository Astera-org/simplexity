import dataclasses
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from penzai import pz
from penzai.toolshed.basic_training import InternalTrainerState, LossFunction

from simplexity.predictive_models.predictive_model import PredictiveModel


def loss_fn(
    model: PredictiveModel,
    state: InternalTrainerState | None,
    rng: chex.PRNGKey,
    inputs: jax.Array,
    labels: jax.Array,
    **kwargs,
) -> tuple[jax.Array, InternalTrainerState | None, dict[str, jax.Array]]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """
    logits = eqx.filter_vmap(model)(inputs)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = jnp.mean(losses)
    return loss, state, {"loss": loss}


def _equinox_trainer_step(
    root_rng: chex.PRNGKey,
    state: InternalTrainerState,
    model: PredictiveModel,
    loss_fn: LossFunction,
    optimizer_def: optax.GradientTransformation,
    kwargs: dict[str, Any],
) -> tuple[
    dict[str, jax.Array],
    Any,
    InternalTrainerState,
]:
    """Implementation of the training step for StatefulTrainer."""
    step_rng = jax.random.fold_in(root_rng, state.step)

    def compute_loss_and_updates(model):
        loss, new_loss_fn_state, aux_outputs = loss_fn(model=model, state=state.loss_fn_state, rng=step_rng, **kwargs)
        return loss, (new_loss_fn_state, aux_outputs)

    grad_fn = jax.grad(compute_loss_and_updates, has_aux=True)
    grads, (new_loss_fn_state, aux_outputs) = grad_fn(model)
    params = eqx.filter(model, eqx.is_array)
    model_updates, new_opt_state = optimizer_def.update(grads, state.opt_state, params)
    return (
        aux_outputs,
        model_updates,
        InternalTrainerState(
            step=state.step + 1,
            opt_state=new_opt_state,
            loss_fn_state=new_loss_fn_state,
        ),
    )


class EquinoxTrainer(eqx.Module):
    """A trainer for a predictive model."""

    root_rng: chex.PRNGKey
    model: pz.StateVariable[PredictiveModel]
    state: pz.StateVariable[InternalTrainerState]
    optimizer_def: optax.GradientTransformation = dataclasses.field(metadata={"pytree_node": False})
    loss_fn: LossFunction = dataclasses.field(metadata={"pytree_node": False})

    @classmethod
    def build(
        cls,
        root_rng: chex.PRNGKey,
        model: PredictiveModel,
        optimizer_def: optax.GradientTransformation,
        loss_fn: LossFunction,
    ) -> "EquinoxTrainer":
        """Build a trainer."""
        params = eqx.filter(model, eqx.is_array)
        initial_opt_state = optimizer_def.init(params)
        state = pz.StateVariable(
            InternalTrainerState(
                step=0,
                opt_state=initial_opt_state,
                loss_fn_state=None,
            ),
            label="EquinoxTrainer.state",
        )
        model_state = pz.StateVariable(model, label="EquinoxTrainer.model")
        return cls(root_rng, model_state, state, optimizer_def, loss_fn)

    def step(self, **kwargs) -> dict[str, jax.Array]:
        """Take a step in training."""
        aux_out, model_updates, new_internal_state = _equinox_trainer_step(
            root_rng=self.root_rng,
            state=self.state.value,
            model=self.model.value,
            loss_fn=self.loss_fn,
            optimizer_def=self.optimizer_def,
            kwargs=kwargs,
        )

        new_model = eqx.apply_updates(self.model.value, model_updates)
        self.model.value = new_model
        self.state.value = new_internal_state

        return aux_out
