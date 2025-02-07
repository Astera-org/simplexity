
import jax.numpy as jnp


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

    m0 = jnp.array([[1], [1], [0]])  # Column vector
    mu0 = jnp.array([[1, -1, -1]])  # Row vector

    def _intermediate_matrix(val):
        return jnp.array([[val, 0, 0], [0, 1, 0], [0, jnp.log(val), 1]])

    transition_matrix = jnp.array([jnp.outer(m0, mu0), _intermediate_matrix(alpha), _intermediate_matrix(beta)])

    # Normalize transition_matrix such that
    # transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum = transition_matrix.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrix_sum)).max()
    transition_matrix = transition_matrix / transition_matrix_sum_max_eigval

    # Verify that transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum_normalized = transition_matrix.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrix_sum_normalized)).max()
    assert jnp.isclose(transition_matrix_sum_max_eigval, 1, atol=1e-10), "Largest absolute eigenvalue is not 1"

    return transition_matrix


def days_of_week():
    """Creates a transition matrix for the Days of the Week Process."""
    transition_matrix = jnp.zeros((11, 7, 7))  # emission, from, to

    d = {"M": 0, "Tu": 1, "W": 2, "Th": 3, "F": 4, "Sa": 5, "Su": 6, "Tmrw": 7, "Yest": 8, "Wknd": 9, "Wkdy": 10}
    all_days = ["M", "Tu", "W", "Th", "F", "Sa", "Su"]
    wkdy_days = [d["M"], d["Tu"], d["W"], d["Th"], d["F"]]
    wknd_days = [d["Sa"], d["Su"]]
    for day in all_days:
        transition_matrix = transition_matrix.at[d[day], :, d[day]].set(1.0)

    for wkdy in wkdy_days:
        transition_matrix = transition_matrix.at[d["Wkdy"], :, wkdy].set(1 / len(wkdy_days))
    for wknd in wknd_days:
        transition_matrix = transition_matrix.at[d["Wknd"], :, wknd].set(1 / len(wknd_days))

    for i, day in enumerate(all_days):
        next_day = all_days[(i + 1) % len(all_days)]
        prev_day = all_days[i - 1]
        transition_matrix = transition_matrix.at[d["Tmrw"], d[day], d[next_day]].set(1.0)
        transition_matrix = transition_matrix.at[d[next_day], d[day], d[next_day]].set(5.0)
        transition_matrix = transition_matrix.at[d["Yest"], d[day], d[prev_day]].set(1.0)

    # normalize so that  transition_matrix.sum(axis=0) is row stochastic
    transition_matrix_sum = transition_matrix.sum(axis=0)
    transition_matrix_row_sums = transition_matrix_sum.sum(axis=0)
    transition_matrix = transition_matrix / transition_matrix_row_sums

    return transition_matrix


def tom_quantum(alpha: float, beta: float):
    """Creates a transition matrix for the Tom Quantum Process."""
    # Common elements
    gamma = 1 / (2 * jnp.sqrt(alpha**2 + beta**2))
    common_diag = 1 / 4
    middle_diag = (alpha**2 - beta**2) * gamma**2
    off_diag = 2 * alpha * beta * gamma**2

    transition_matrix = jnp.array(
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

    return transition_matrix


def fanizza(alpha: float, lamb: float):
    """Creates a transition matrix for the Faniza Process."""
    # Calculate intermediate values
    a_la = (1 - lamb * jnp.cos(alpha) + lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)
    b_la = (1 - lamb * jnp.cos(alpha) - lamb * jnp.sin(alpha)) / (1 - 2 * lamb * jnp.cos(alpha) + lamb**2)

    # Define tau
    tau = jnp.ones(4)

    # Define the reset distribution pi0
    pi0 = jnp.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])

    # Define w
    w = jnp.array(
        [1, 1 - lamb, 1 + lamb * (jnp.sin(alpha) - jnp.cos(alpha)), 1 - lamb * (jnp.sin(alpha) + jnp.cos(alpha))]
    )

    # Create Da
    Da = jnp.outer(w, pi0)

    # Create Db (with the sine sign error fixed)
    Db = lamb * jnp.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, jnp.cos(alpha), -jnp.sin(alpha)], [0, 0, jnp.sin(alpha), jnp.cos(alpha)]]
    )

    transition_matrix = jnp.stack([Da, Db], axis=0)

    # Verify that transition_matrix @ tau = tau (stochasticity condition)
    assert jnp.allclose(transition_matrix[0] @ tau + transition_matrix[1] @ tau, tau), "Stochasticity condition not met"

    return transition_matrix


def rrxor(pR1=0.5, pR2=0.5):
    """Creates a transition matrix for the RRXOR Process."""
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
    transition_matrix = jnp.zeros((2, 5, 5))
    transition_matrix = transition_matrix.at[0, s["S"], s["0"]].set(pR1)
    transition_matrix = transition_matrix.at[1, s["S"], s["1"]].set(1 - pR1)
    transition_matrix = transition_matrix.at[0, s["0"], s["F"]].set(pR2)
    transition_matrix = transition_matrix.at[1, s["0"], s["T"]].set(1 - pR2)
    transition_matrix = transition_matrix.at[0, s["1"], s["T"]].set(pR2)
    transition_matrix = transition_matrix.at[1, s["1"], s["F"]].set(1 - pR2)
    transition_matrix = transition_matrix.at[1, s["T"], s["S"]].set(1.0)
    transition_matrix = transition_matrix.at[0, s["F"], s["S"]].set(1.0)
    return transition_matrix


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
