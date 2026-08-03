from typing import NamedTuple

import jax
import jax.numpy as jnp

from tests.vendored_process.generators_copy.utils import principal_ev


class Data(NamedTuple):
    Ts: jax.Array
    eta_0: jax.Array
    w: jax.Array


def validate(Ts: jax.Array) -> bool:
    if len(Ts.shape) != 3:
        return False
    if any(dim == 0 for dim in Ts.shape):
        return False
    if Ts.shape[1] != Ts.shape[2]:
        return False
    if not jnp.all(jnp.isfinite(Ts)):
        return False
    T = jnp.sum(Ts, axis=0)
    norm = jnp.linalg.norm(T, ord=jnp.inf)
    return bool(jnp.isclose(norm, 1))


def init(Ts: jax.Array) -> Data:
    T = Ts.sum(axis=0)
    w = principal_ev(T)
    eta_0 = principal_ev(T.T)
    eta_0 /= eta_0 @ w
    return Data(Ts=Ts, eta_0=eta_0, w=w)


def obs_dist(data: Data, eta: jax.Array) -> jax.Array:
    return eta @ data.Ts @ data.w


def sample(data: Data, eta: jax.Array, key: jax.Array) -> jax.Array:
    probs = obs_dist(data, eta)
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)
    return jax.random.categorical(key, logits)


def update(data: Data, eta: jax.Array, x: jax.Array) -> jax.Array:
    eta = eta @ data.Ts[x]
    return eta / (eta @ data.w)


def generate(data: Data, eta: jax.Array, keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    def fn(eta: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = sample(data, eta, key)
        eta = update(data, eta, x)
        return eta, x

    return jax.lax.scan(fn, eta, keys)


def seq_prob(data: Data, xs: jax.Array) -> jax.Array:
    def fn(eta, x):
        return eta @ data.Ts[x], None

    eta, _ = jax.lax.scan(fn, init=data.eta_0, xs=xs)
    return eta @ data.w
