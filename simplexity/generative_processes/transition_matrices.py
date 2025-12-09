"""Transition matrices for generative processes."""

import jax
import jax.numpy as jnp


def get_stationary_state(state_transition_matrix: jax.Array) -> jax.Array:
    """Compute the stationary distribution of a transition matrix."""
    eigenvalues, left_eigenvectors = jnp.linalg.eig(state_transition_matrix)
    stationary_state = left_eigenvectors[:, jnp.isclose(eigenvalues, 1)].real
    assert stationary_state.shape == (state_transition_matrix.shape[1], 1)
    return stationary_state.squeeze(axis=-1) / jnp.sum(stationary_state)


def coin(p: float):
    """Create a transition matrix for a simple coin-flip Process."""
    return jnp.array([[[p]], [[1 - p]]])


def days_of_week() -> jax.Array:
    """Creates a transition matrix for the Days of the Week Process."""
    d = {"M": 0, "Tu": 1, "W": 2, "Th": 3, "F": 4, "Sa": 5, "Su": 6, "Tmrw": 7, "Yest": 8, "Wknd": 9, "Wkdy": 10}
    all_days = ["M", "Tu", "W", "Th", "F", "Sa", "Su"]
    wkdy_days = [d["M"], d["Tu"], d["W"], d["Th"], d["F"]]
    wknd_days = [d["Sa"], d["Su"]]

    transition_matrices = jnp.zeros((11, 7, 7))

    for day in all_days:
        transition_matrices = transition_matrices.at[d[day], :, d[day]].set(1.0)

    for wkdy in wkdy_days:
        transition_matrices = transition_matrices.at[d["Wkdy"], :, wkdy].set(1 / len(wkdy_days))

    for wknd in wknd_days:
        transition_matrices = transition_matrices.at[d["Wknd"], :, wknd].set(1 / len(wknd_days))

    for i, day in enumerate(all_days):
        next_day = all_days[(i + 1) % len(all_days)]
        prev_day = all_days[i - 1]
        transition_matrices = transition_matrices.at[d["Tmrw"], d[day], d[next_day]].set(1.0)
        transition_matrices = transition_matrices.at[d[next_day], d[day], d[next_day]].set(5.0)
        transition_matrices = transition_matrices.at[d["Yest"], d[day], d[prev_day]].set(1.0)

    transition_matrices = transition_matrices / transition_matrices.sum(axis=(0, 2), keepdims=True)

    return transition_matrices


def even_ones(p: float) -> jax.Array:
    """Creates a transition matrix for the Even Ones Process.

    Defined in:  https://arxiv.org/pdf/1412.2859 Fig 3. using p = 0.5
    Steady-state distribution = [2, 1] / 3
    """
    assert 0 <= p <= 1
    q = 1 - p
    return jnp.array(
        [
            [
                [q, 0],
                [0, 0],
            ],
            [
                [0, p],
                [1, 0],
            ],
        ]
    )


def fanizza(alpha: float, lamb: float) -> jax.Array:
    """Creates a transition matrix for the Faniza Process."""
    a_la = (1 - lamb * jnp.cos(alpha) + lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)
    b_la = (1 - lamb * jnp.cos(alpha) - lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)

    # Define the reset distribution
    pi0 = jnp.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])

    w = jnp.array(
        [1, 1 - lamb, 1 + lamb * (jnp.sin(alpha) - jnp.cos(alpha)), 1 - lamb * (jnp.sin(alpha) + jnp.cos(alpha))]
    )

    da = jnp.outer(w, pi0)

    db = lamb * jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, jnp.cos(alpha), -jnp.sin(alpha)],
            [0, 0, jnp.sin(alpha), jnp.cos(alpha)],
        ]
    )

    return jnp.stack([da, db], axis=0)


def leaky_rrxor(pR1: float, pR2: float, epsilon: float) -> jax.Array:
    """Creates a transition matrix for the leaky RRXOR Process."""
    assert 0 <= epsilon <= 1

    transition_matrices_base = rrxor(pR1, pR2)
    leak = jnp.ones((2, 5, 5))

    transition_matrices = (1 - epsilon) * transition_matrices_base + (epsilon / 10) * leak

    return transition_matrices


def matching_parens(open_probs: list[float]) -> jax.Array:
    """Creates a model for generating Matching Parentheses."""
    if len(open_probs) < 1:
        raise TypeError("Must provide a list of at least one open_probability")
    if any(p <= 0 or p > 1 for p in open_probs):
        raise TypeError(f"`open_probs` elements must all be in (0, 1].  Got: {open_probs}")
    if open_probs[0] != 1.0:
        raise ValueError("First open_prob must equal 1.0")
    if open_probs[-1] != 0.0:
        open_probs = open_probs + [0.0]
    prob_array = jnp.array(open_probs)
    return jnp.stack([jnp.diag(prob_array[:-1], k=1), jnp.diag(1.0 - prob_array[1:], k=-1)])


def mess3(x: float, a: float) -> jax.Array:
    """Creates a transition matrix for the Mess3 Process."""
    b = (1 - a) / 2
    y = 1 - 2 * x

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    return jnp.array(
        [
            [
                [ay, bx, bx],
                [ax, by, bx],
                [ax, bx, by],
            ],
            [
                [by, ax, bx],
                [bx, ay, bx],
                [bx, ax, by],
            ],
            [
                [by, bx, ax],
                [bx, by, ax],
                [bx, bx, ay],
            ],
        ]
    )


def mr_name(p: float, q: float) -> jax.Array:
    """Creates a transition matrix for the Mr. Dursley/Wonka Process."""
    assert 0 <= p <= 1
    assert 0 <= q <= 1
    assert 0 <= 1 - p - q <= 1
    vocab = ["Mr.", "Something", "Blah", "name"]
    d = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    num_states = len(vocab)
    transition_matrices = jnp.zeros((vocab_size, num_states, num_states))
    transition_matrices = transition_matrices.at[
        d["name"],
        d["Mr."],
        d["name"],
    ].set(1)
    transition_matrices = transition_matrices.at[
        d["Something"],
        d["name"],
        d["Something"],
    ].set(0.5)
    transition_matrices = transition_matrices.at[
        d["Blah"],
        d["name"],
        d["Blah"],
    ].set(0.5)
    transition_matrices = transition_matrices.at[
        d["Mr."],
        d["Something"],
        d["Mr."],
    ].set(q)
    transition_matrices = transition_matrices.at[
        d["Something"],
        d["Something"],
        d["Something"],
    ].set(1 - p - q)
    transition_matrices = transition_matrices.at[
        d["Blah"],
        d["Something"],
        d["Blah"],
    ].set(p)
    transition_matrices = transition_matrices.at[
        d["Mr."],
        d["Blah"],
        d["Mr."],
    ].set(q)
    transition_matrices = transition_matrices.at[
        d["Something"],
        d["Blah"],
        d["Something"],
    ].set(p)
    transition_matrices = transition_matrices.at[
        d["Blah"],
        d["Blah"],
        d["Blah"],
    ].set(1 - p - q)
    return transition_matrices


def no_consecutive_ones(p: float) -> jax.Array:
    """Creates a transition matrix for the No Consecutive Ones Process.

    Steady-state distribution = [2, 1] / 3
    """
    assert 0 <= p <= 1
    q = 1 - p
    return jnp.array(
        [
            [
                [q, 0],
                [1, 0],
            ],
            [
                [0, p],
                [0, 0],
            ],
        ]
    )


def _validate_post_quantum_conditions(alpha: jax.Array, beta: float) -> None:
    if not alpha > 1 > beta > 0:
        raise ValueError("Condition alpha > 1 > beta > 0 not satisfied")
    if alpha + beta == 2:
        raise ValueError("Condition alpha + beta ≠ 2 not satisfied")
    if jnp.isclose(jnp.log(alpha) / jnp.log(beta), jnp.round(jnp.log(alpha) / jnp.log(beta))):
        raise ValueError("Condition ln(alpha) / ln(beta) ∉ ℚ not satisfied")


def post_quantum(log_alpha: float, beta: float) -> jax.Array:
    """Creates a transition matrix for the Post Quantum Process."""
    alpha = jnp.exp(log_alpha)
    _validate_post_quantum_conditions(alpha, beta)

    m0 = jnp.array([1, 1, 0])
    mu0 = jnp.array([1, -1, -1])

    def _intermediate_matrix(val):
        return jnp.array(
            [
                [val, 0, 0],
                [0, 1, 0],
                [0, jnp.log(val), 1],
            ]
        )

    transition_matrices = jnp.array([jnp.outer(m0, mu0), _intermediate_matrix(alpha), _intermediate_matrix(beta)])

    # Normalize transition_matrices such that
    # transition_matrices[0] + transition_matrices[1] + transition_matrices[2] has largest abs eigenvalue = 1
    transition_matrices_sum = transition_matrices.sum(axis=0)
    transition_matrices_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrices_sum)).max()
    transition_matrices = transition_matrices / transition_matrices_sum_max_eigval

    return transition_matrices


def rrxor(p1: float, p2: float) -> jax.Array:
    """Creates a transition matrix for the RRXOR Process.

    Steady-state distribution = [2, 1, 1, 1, 1] / 6
    """
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}

    transition_matrices = jnp.zeros((2, 5, 5))
    transition_matrices = transition_matrices.at[0, s["S"], s["0"]].set(p1)
    transition_matrices = transition_matrices.at[1, s["S"], s["1"]].set(1 - p1)
    transition_matrices = transition_matrices.at[0, s["0"], s["F"]].set(p2)
    transition_matrices = transition_matrices.at[1, s["0"], s["T"]].set(1 - p2)
    transition_matrices = transition_matrices.at[0, s["1"], s["T"]].set(p2)
    transition_matrices = transition_matrices.at[1, s["1"], s["F"]].set(1 - p2)
    transition_matrices = transition_matrices.at[1, s["T"], s["S"]].set(1.0)
    transition_matrices = transition_matrices.at[0, s["F"], s["S"]].set(1.0)

    return transition_matrices


def sns(p: float, q: float):
    """Creates a transition matrix for the Simple Nonunifilar Source Process.

    Defined in https://arxiv.org/pdf/1702.08565 Fig 2.
    """
    return jnp.array(
        [
            [
                [1 - p, p],
                [0, 1 - q],
            ],
            [
                [0, 0],
                [q, 0],
            ],
        ]
    )


def tom_quantum(alpha: float, beta: float) -> jax.Array:
    """Creates a transition matrix for the Tom Quantum Process."""
    gamma2 = 1 / (4 * (alpha**2 + beta**2))
    common_diag = 1 / 4
    middle_diag = (alpha**2 - beta**2) * gamma2
    off_diag = 2 * alpha * beta * gamma2

    transition_matrices = jnp.array(
        [
            [
                [common_diag, 0, off_diag],
                [0, middle_diag, 0],
                [off_diag, 0, common_diag],
            ],
            [
                [common_diag, 0, -off_diag],
                [0, middle_diag, 0],
                [-off_diag, 0, common_diag],
            ],
            [
                [common_diag, off_diag, 0],
                [off_diag, common_diag, 0],
                [0, 0, middle_diag],
            ],
            [
                [common_diag, -off_diag, 0],
                [-off_diag, common_diag, 0],
                [0, 0, middle_diag],
            ],
        ]
    )

    return transition_matrices


def zero_one_random(p: float) -> jax.Array:
    """Creates a transition matrix for the Zero One Random (Z1R) Process.

    Steady-state distribution = [1, 1, 1] / 3
    """
    assert 0 <= p <= 1
    q = 1 - p
    return jnp.array(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [q, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [p, 0, 0],
            ],
        ]
    )


HMM_MATRIX_FUNCTIONS = {
    "coin": coin,
    "days_of_week": days_of_week,
    "even_ones": even_ones,
    "leaky_rrxor": leaky_rrxor,
    "matching_parens": matching_parens,
    "mess3": mess3,
    "mr_name": mr_name,
    "no_consecutive_ones": no_consecutive_ones,
    "rrxor": rrxor,
    "sns": sns,
    "zero_one_random": zero_one_random,
}

GHMM_MATRIX_FUNCTIONS = HMM_MATRIX_FUNCTIONS | {
    "fanizza": fanizza,
    "post_quantum": post_quantum,
    "tom_quantum": tom_quantum,
}
