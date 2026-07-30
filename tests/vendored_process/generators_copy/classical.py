import jax
import jax.numpy as jnp


def _mess_trans(x: float, s: int) -> jax.Array:
    r"""State transition matrix for Mess process.

    Args:
        x: Transition probability to each other state.
            x = P(s_t = s' \forall s' \in {0, ..., s-1} \ s')
            y: Probability of staying in the same state.
            y = P(s_t = s_t-1)
        s: Number of states

    Returns:
        State transition matrix
            T[i, j] = P(s_t = i | s_{t-1} = j)
            T[i, i] = y
            T[i, j] = x for i != j
    """
    assert x >= 0
    assert x <= 1
    y = 1 - (s - 1) * x
    assert y >= 0
    assert y <= 1
    return (x * (jnp.ones((s, s)) - jnp.eye(s))) + (y * jnp.eye(s))


def _mess_emit(a: float, s: int) -> jax.Array:
    r"""Emission matrix for Mess process.

    Args:
        a: Emission probability corresponding to the previous state.
            a = P(x_t = s_t-1)
            b: Emission probabilities for each other value.
            b = P(x_t = s' \forall s' \in S_{t-1})
        s: Number of states
        rd: Ratio difference
            If the vocab size to state size ratio, V : S
            is reduced, num : denom (V / S = num / denom)
            rd := num - denom

    Returns:
        Emission matrix
            Tv[i, j] = P(x_t = j | s_t = i)
            Tv[i, i] = a
            Tv[i, j] = b for i != j
    """
    assert a >= 0
    assert a <= 1
    b = (1 - a) / (s - 1)

    return (a * jnp.eye(s)) + (b * (jnp.ones((s, s)) - jnp.eye(s)))


def mess(x: float, a: float, s: int) -> jax.Array:
    r"""Mess process.

    Args:
        x: Transition probability to each other state.
        a: Emission probability corresponding to the previous state.
        s: Number of states
        rd: Ratio difference

    Returns:
        Transition matrix
            Ts[o, j, k] = P(x_t=o, s_t=k | s_{t-1}=j)
    """
    assert s > 0
    T_trans = _mess_trans(x, s)
    T_emit = _mess_emit(a, s)

    # Ts[o, j, k] = P(x_t=o, s_t=k | s_{t-1}=j) = Tv[k, o] * T[k, j]
    inner = T_emit[:, :, None] * T_trans[None, :, :]
    return jnp.transpose(inner, (0, 2, 1))
