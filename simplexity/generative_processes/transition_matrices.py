import jax.numpy as jnp


def no_consecutive_ones(p: float = 0.5):
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


def even_ones(p: float = 0.5):
    """Creates a transition matrix for the Even Ones Process.

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


def zero_one_random(p: float = 0.5):
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


def _validate_post_quantum_conditions(alpha, beta):
    if not (alpha > 1 > beta > 0):
        raise ValueError("Condition alpha > 1 > beta > 0 not satisfied")
    if alpha + beta == 2:
        raise ValueError("Condition alpha + beta ≠ 2 not satisfied")
    if jnp.isclose(jnp.log(alpha) / jnp.log(beta), jnp.round(jnp.log(alpha) / jnp.log(beta))):
        raise ValueError("Condition ln(alpha) / ln(beta) ∉ ℚ not satisfied")


def post_quantum(log_alpha=1, beta=0.5):
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


def days_of_week():
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


def tom_quantum(alpha: float, beta: float):
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


def fanizza(alpha: float, lamb: float):
    """Creates a transition matrix for the Faniza Process."""
    a_la = (1 - lamb * jnp.cos(alpha) + lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)
    b_la = (1 - lamb * jnp.cos(alpha) - lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)

    # Define the reset distribution
    pi0 = jnp.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])

    w = jnp.array(
        [1, 1 - lamb, 1 + lamb * (jnp.sin(alpha) - jnp.cos(alpha)), 1 - lamb * (jnp.sin(alpha) + jnp.cos(alpha))]
    )

    Da = jnp.outer(w, pi0)

    Db = lamb * jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, jnp.cos(alpha), -jnp.sin(alpha)],
            [0, 0, jnp.sin(alpha), jnp.cos(alpha)],
        ]
    )

    return jnp.stack([Da, Db], axis=0)


def rrxor(pR1=0.5, pR2=0.5):
    """Creates a transition matrix for the RRXOR Process.

    Steady-state distribution = [2, 1, 1, 1, 1] / 6
    """
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}

    transition_matrices = jnp.zeros((2, 5, 5))
    transition_matrices = transition_matrices.at[0, s["S"], s["0"]].set(pR1)
    transition_matrices = transition_matrices.at[1, s["S"], s["1"]].set(1 - pR1)
    transition_matrices = transition_matrices.at[0, s["0"], s["F"]].set(pR2)
    transition_matrices = transition_matrices.at[1, s["0"], s["T"]].set(1 - pR2)
    transition_matrices = transition_matrices.at[0, s["1"], s["T"]].set(pR2)
    transition_matrices = transition_matrices.at[1, s["1"], s["F"]].set(1 - pR2)
    transition_matrices = transition_matrices.at[1, s["T"], s["S"]].set(1.0)
    transition_matrices = transition_matrices.at[0, s["F"], s["S"]].set(1.0)

    return transition_matrices


def mess3(x=0.15, a=0.6):
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
