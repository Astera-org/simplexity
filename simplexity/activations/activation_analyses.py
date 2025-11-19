"""Analysis implementations for Transformer layer activations."""

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from sklearn.linear_model import LinearRegression


def _compute_pca(
    X: JaxArray,
    n_components: int | None = None,
    weights: JaxArray | None = None,
    center: bool = True,
) -> dict[str, JaxArray]:
    """Compute weighted PCA via eigendecomposition of covariance matrix."""
    N, D = X.shape

    if N == 0 or D == 0:
        raise ValueError("Cannot compute PCA on empty data")

    if n_components is None:
        n_components = min(N, D)
    n_components = min(n_components, min(N, D))

    if weights is None:
        w = None
        mean = X.mean(axis=0) if center else jnp.zeros(D, dtype=X.dtype)
    else:
        w = weights.astype(float)
        if w.sum() <= 0:
            raise ValueError("Sum of weights must be positive")
        if w.ndim != 1 or w.shape[0] != N:
            raise ValueError(f"Weights must be shape (N,), got {w.shape} for N={N}")
        total = w.sum()
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        w = w / total

        if center:
            mean = jnp.average(X, axis=0, weights=w)
        else:
            mean = jnp.zeros(D, dtype=X.dtype)

    Xc = X - mean

    if w is None:
        cov = (Xc.T @ Xc) / Xc.shape[0]
    else:
        cov = (Xc * w[:, None]).T @ Xc

    eigvals, eigvecs = jnp.linalg.eigh(cov)

    idx = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_sel = eigvals[:n_components]
    eigvecs_sel = eigvecs[:, :n_components]

    total_var = eigvals.sum()
    if total_var <= 0:
        explained_ratio = jnp.zeros_like(eigvals_sel)
        all_explained_ratio = jnp.zeros_like(eigvals)
    else:
        explained_ratio = eigvals_sel / total_var
        all_explained_ratio = eigvals / total_var

    X_proj = Xc @ eigvecs_sel

    return {
        "components": eigvecs_sel.T,
        "explained_variance": eigvals_sel,
        "explained_variance_ratio": explained_ratio,
        "mean": mean,
        "X_proj": X_proj,
        "all_explained_variance": eigvals,
        "all_explained_variance_ratio": all_explained_ratio,
    }


def _compute_variance_thresholds(
    all_explained_variance_ratio: JaxArray,
    thresholds: Sequence[float],
) -> dict[float, int]:
    """Compute number of components needed to reach variance thresholds."""
    cum_var = jnp.cumsum(all_explained_variance_ratio)
    result = {}

    for threshold in thresholds:
        indices = jnp.where(cum_var >= threshold)[0]
        if len(indices) > 0:
            result[threshold] = int(indices[0]) + 1
        else:
            result[threshold] = len(cum_var)

    return result


class ActivationAnalysis(Protocol):
    """Protocol for activation analysis implementations."""

    _requires_belief_states: bool
    _token_selection: Literal["all", "last"]
    _concat_layers: bool
    _use_probs_as_weights: bool

    def analyze(
        self,
        activations: Mapping[str, JaxArray],
        weights: JaxArray,
        belief_states: JaxArray | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, JaxArray]]:
        """Analyze activations and return scalar metrics and projections."""
        ...


class PCAAnalysis:
    """Weighted principal component analysis of layer activations."""

    _requires_belief_states: bool = False
    _token_selection: Literal["all", "last"]
    _concat_layers: bool
    _use_probs_as_weights: bool

    def __init__(
        self,
        n_components: int | None = None,
        variance_thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
        token_selection: Literal["all", "last"] = "all",
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
    ):
        """Initialize PCA analysis."""
        self._n_components = n_components
        self._variance_thresholds = variance_thresholds
        self._token_selection = token_selection
        self._concat_layers = concat_layers
        self._use_probs_as_weights = use_probs_as_weights

    def analyze(
        self,
        activations: Mapping[str, JaxArray],
        weights: JaxArray,
        belief_states: JaxArray | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, JaxArray]]:
        """Perform weighted PCA and return variance metrics and projections."""
        scalars = {}
        projections = {}

        for layer_name, layer_acts in activations.items():
            pca_result = _compute_pca(
                layer_acts,
                n_components=self._n_components,
                weights=weights,
                center=True,
            )

            cumulative_variance = jnp.cumsum(pca_result["explained_variance_ratio"])
            for i, cumvar in enumerate(cumulative_variance, start=1):
                scalars[f"{layer_name}_cumvar_{i}"] = float(cumvar)

            scalars[f"{layer_name}_variance_explained"] = float(cumulative_variance[-1])

            threshold_counts = _compute_variance_thresholds(
                pca_result["all_explained_variance_ratio"],
                self._variance_thresholds,
            )

            for threshold, n_comps in threshold_counts.items():
                threshold_pct = int(threshold * 100)
                scalars[f"{layer_name}_n_components_{threshold_pct}pct"] = float(n_comps)

            projections[f"{layer_name}_pca"] = pca_result["X_proj"]

        return scalars, projections


class LinearRegressionAnalysis:
    """Weighted linear regression from activations to belief states using sklearn."""

    _requires_belief_states: bool = True
    _token_selection: Literal["all", "last"]
    _concat_layers: bool
    _use_probs_as_weights: bool

    def __init__(
        self,
        token_selection: Literal["all", "last"] = "all",
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
    ):
        """Initialize linear regression analysis."""
        self._token_selection = token_selection
        self._concat_layers = concat_layers
        self._use_probs_as_weights = use_probs_as_weights

    def analyze(
        self,
        activations: Mapping[str, JaxArray],
        weights: JaxArray,
        belief_states: JaxArray | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, JaxArray]]:
        """Fit weighted linear regression and return metrics."""
        if belief_states is None:
            raise ValueError("LinearRegressionAnalysis requires belief_states")

        belief_states_np = np.asarray(belief_states)
        weights_np = np.asarray(weights)

        scalars = {}
        projections = {}

        for layer_name, layer_acts in activations.items():
            X = np.asarray(layer_acts)
            Y = belief_states_np

            model = LinearRegression()
            model.fit(X, Y, sample_weight=weights_np)

            Y_pred = model.predict(X)
            residuals = Y_pred - Y

            weighted_sq_residuals = (residuals**2) * weights_np[:, np.newaxis]
            mse = weighted_sq_residuals.mean(axis=0)
            mae = (np.abs(residuals) * weights_np[:, np.newaxis]).mean(axis=0)
            rmse = np.sqrt(mse)

            weighted_ss_res = weighted_sq_residuals.sum()
            y_mean = np.average(Y, axis=0, weights=weights_np)
            weighted_ss_tot = ((Y - y_mean) ** 2 * weights_np[:, np.newaxis]).sum()
            r2 = 1 - (weighted_ss_res / weighted_ss_tot) if weighted_ss_tot > 0 else 0.0

            dists = np.sqrt(np.sum(residuals**2, axis=1))
            dist = float((dists * weights_np).sum())

            scalars[f"{layer_name}_r2"] = float(r2)
            scalars[f"{layer_name}_rmse"] = float(rmse.mean())
            scalars[f"{layer_name}_mae"] = float(mae.mean())
            scalars[f"{layer_name}_dist"] = dist

            projections[f"{layer_name}_projected"] = jnp.asarray(Y_pred)

        return scalars, projections


class LinearRegressionSVDAnalysis:
    """Weighted linear regression from activations to belief states with SVD and rcond tuning."""

    _requires_belief_states: bool = True
    _token_selection: Literal["all", "last"]
    _concat_layers: bool
    _use_probs_as_weights: bool

    def __init__(
        self,
        token_selection: Literal["all", "last"] = "all",
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
        rcond_values: Sequence[float] | None = None,
    ):
        """Initialize SVD linear regression analysis."""
        self._token_selection = token_selection
        self._concat_layers = concat_layers
        self._use_probs_as_weights = use_probs_as_weights
        self._rcond_values = rcond_values if rcond_values else [1e-15]

    def analyze(
        self,
        activations: Mapping[str, JaxArray],
        weights: JaxArray,
        belief_states: JaxArray | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, JaxArray]]:
        """Fit weighted linear regression with SVD-based rcond tuning."""
        if belief_states is None:
            raise ValueError("LinearRegressionSVDAnalysis requires belief_states")

        scalars = {}
        projections = {}

        for layer_name, layer_acts in activations.items():
            X = layer_acts
            Y = belief_states
            N, D = X.shape

            X_bias = jnp.concatenate([jnp.ones((N, 1)), X], axis=1)

            sqrt_w = jnp.sqrt(weights).reshape(-1, 1)
            X_weighted = X_bias * sqrt_w
            Y_weighted = Y * sqrt_w

            U, S, Vh = jnp.linalg.svd(X_weighted, full_matrices=False)
            max_singular_value = S[0]

            best_error = float("inf")
            best_rcond = self._rcond_values[0]
            best_beta = None

            for rcond in self._rcond_values:
                threshold = rcond * max_singular_value

                S_pinv = jnp.zeros_like(S)
                S_pinv = jnp.where(S > threshold, 1.0 / S, 0.0)

                pinv_matrix = Vh.T @ jnp.diag(S_pinv) @ U.T
                beta = pinv_matrix @ Y_weighted

                Y_pred = X_bias @ beta
                residuals = Y_pred - Y
                dists = jnp.sqrt(jnp.sum(residuals**2, axis=1))
                error = (dists * weights).sum()

                if error < best_error:
                    best_error = error
                    best_rcond = rcond
                    best_beta = beta

            if best_beta is None:
                raise RuntimeError("Failed to find valid regression solution")
            Y_pred = X_bias @ best_beta
            residuals = Y_pred - Y

            weighted_sq_residuals = (residuals**2) * weights[:, jnp.newaxis]
            mse = weighted_sq_residuals.mean(axis=0)
            mae = (jnp.abs(residuals) * weights[:, jnp.newaxis]).mean(axis=0)
            rmse = jnp.sqrt(mse)

            weighted_ss_res = weighted_sq_residuals.sum()
            y_mean = jnp.average(Y, axis=0, weights=weights)
            weighted_ss_tot = ((Y - y_mean) ** 2 * weights[:, jnp.newaxis]).sum()
            r2 = 1 - (weighted_ss_res / weighted_ss_tot) if weighted_ss_tot > 0 else 0.0

            dists = jnp.sqrt(jnp.sum(residuals**2, axis=1))
            dist = float((dists * weights).sum())

            scalars[f"{layer_name}_r2"] = float(r2)
            scalars[f"{layer_name}_rmse"] = float(rmse.mean())
            scalars[f"{layer_name}_mae"] = float(mae.mean())
            scalars[f"{layer_name}_dist"] = dist
            scalars[f"{layer_name}_best_rcond"] = float(best_rcond)

            projections[f"{layer_name}_projected"] = Y_pred

        return scalars, projections


ALL_ANALYSES = {
    "pca": PCAAnalysis,
    "linear_regression": LinearRegressionAnalysis,
    "linear_regression_svd": LinearRegressionSVDAnalysis,
}
