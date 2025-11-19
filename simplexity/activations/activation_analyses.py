"""Analysis implementations for Transformer layer activations."""

from collections.abc import Mapping, Sequence

import jax
import numpy as np
from sklearn.linear_model import LinearRegression


def _to_numpy(x: jax.Array | np.ndarray) -> np.ndarray:
    """Convert JAX or NumPy array to NumPy."""
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return np.asarray(x)


def _compute_pca(
    X: np.ndarray,
    n_components: int | None = None,
    weights: np.ndarray | None = None,
    center: bool = True,
) -> dict[str, np.ndarray]:
    """Compute weighted PCA via eigendecomposition of covariance matrix."""
    X = _to_numpy(X)
    N, D = X.shape

    if n_components is None:
        n_components = min(N, D)
    n_components = min(n_components, min(N, D))

    if weights is None:
        w = None
        mean = X.mean(axis=0) if center else np.zeros(D, dtype=X.dtype)
    else:
        w = _to_numpy(weights).astype(float)
        if w.ndim != 1 or w.shape[0] != N:
            raise ValueError(f"Weights must be shape (N,), got {w.shape} for N={N}")
        total = w.sum()
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        w = w / total

        if center:
            mean = np.average(X, axis=0, weights=w)
        else:
            mean = np.zeros(D, dtype=X.dtype)

    Xc = X - mean

    if w is None:
        cov = (Xc.T @ Xc) / Xc.shape[0]
    else:
        cov = (Xc * w[:, None]).T @ Xc

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_sel = eigvals[:n_components]
    eigvecs_sel = eigvecs[:, :n_components]

    total_var = eigvals.sum()
    if total_var <= 0:
        explained_ratio = np.zeros_like(eigvals_sel)
        all_explained_ratio = np.zeros_like(eigvals)
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
    all_explained_variance_ratio: np.ndarray,
    thresholds: Sequence[float],
) -> dict[float, int]:
    """Compute number of components needed to reach variance thresholds."""
    cum_var = np.cumsum(all_explained_variance_ratio)
    result = {}

    for threshold in thresholds:
        indices = np.where(cum_var >= threshold)[0]
        if len(indices) > 0:
            result[threshold] = int(indices[0]) + 1
        else:
            result[threshold] = len(cum_var)

    return result


class PCAAnalysis:
    """Weighted principal component analysis of layer activations."""

    _requires_belief_states = False

    def __init__(
        self,
        n_components: int | None = None,
        variance_thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
        token_selection: str = "all",
        layer_selection: str = "individual",
        use_probs_as_weights: bool = True,
    ):
        """Initialize PCA analysis."""
        self._n_components = n_components
        self._variance_thresholds = variance_thresholds
        self._token_selection: str = token_selection
        self._layer_selection: str = layer_selection
        self._use_probs_as_weights = use_probs_as_weights

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Perform weighted PCA and return variance metrics and projections."""
        weights_np = np.asarray(weights)

        scalars = {}
        projections = {}

        for layer_name, layer_acts in activations.items():
            X = np.asarray(layer_acts)

            pca_result = _compute_pca(
                X,
                n_components=self._n_components,
                weights=weights_np,
                center=True,
            )

            cumulative_variance = np.cumsum(pca_result["explained_variance_ratio"])
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

            projections[f"{layer_name}_pca"] = jax.numpy.asarray(pca_result["X_proj"])

        return scalars, projections


class LinearRegressionAnalysis:
    """Weighted linear regression from activations to belief states with rcond tuning."""

    _requires_belief_states = True

    def __init__(
        self,
        token_selection: str = "all",
        layer_selection: str = "individual",
        use_probs_as_weights: bool = True,
        rcond_values: Sequence[float] | None = None,
    ):
        """Initialize linear regression analysis."""
        self._token_selection = token_selection
        self._layer_selection = layer_selection
        self._use_probs_as_weights = use_probs_as_weights
        self._rcond_values = rcond_values if rcond_values is not None else [1e-15]

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Fit weighted linear regression with SVD-based rcond tuning."""
        if belief_states is None:
            raise ValueError("LinearRegressionAnalysis requires belief_states")

        belief_states_np = np.asarray(belief_states)
        weights_np = np.asarray(weights)

        # Normalize weights to sum to 1
        weights_np = weights_np / weights_np.sum()

        scalars = {}
        projections = {}

        for layer_name, layer_acts in activations.items():
            X = np.asarray(layer_acts)  # (N, D)
            Y = belief_states_np  # (N, B)
            N, D = X.shape

            # Add bias column
            X_bias = np.concatenate([np.ones((N, 1)), X], axis=1)  # (N, D+1)

            # Weighted design matrices (using sqrt of weights)
            sqrt_w = np.sqrt(weights_np).reshape(-1, 1)  # (N, 1)
            X_weighted = X_bias * sqrt_w  # (N, D+1)
            Y_weighted = Y * sqrt_w  # (N, B)

            # Compute SVD once for all rcond values
            U, S, Vh = np.linalg.svd(X_weighted, full_matrices=False)
            max_singular_value = S[0]

            # Sweep over rcond values to find best regularization
            best_error = float("inf")
            best_rcond = self._rcond_values[0]
            best_beta = None

            for rcond in self._rcond_values:
                # Compute threshold for this rcond value
                threshold = rcond * max_singular_value

                # Create reciprocal of singular values with thresholding
                S_pinv = np.zeros_like(S)
                above_threshold = S > threshold
                S_pinv[above_threshold] = 1.0 / S[above_threshold]

                # Compute pseudoinverse
                pinv_matrix = Vh.T @ np.diag(S_pinv) @ U.T  # (D+1, N)

                # Solve: beta = pinv(X_weighted) @ Y_weighted
                beta = pinv_matrix @ Y_weighted  # (D+1, B)

                # Compute predictions and error
                Y_pred = X_bias @ beta  # (N, B)
                residuals = Y_pred - Y
                dists = np.sqrt(np.sum(residuals**2, axis=1))
                error = (dists * weights_np).sum()

                if error < best_error:
                    best_error = error
                    best_rcond = rcond
                    best_beta = beta

            # Compute final predictions with best beta
            if best_beta is None:
                raise RuntimeError("Failed to find valid regression solution")
            Y_pred = X_bias @ best_beta  # (N, B)
            residuals = Y_pred - Y

            # Compute metrics
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
            scalars[f"{layer_name}_best_rcond"] = float(best_rcond)

            projections[f"{layer_name}_projected"] = jax.numpy.asarray(Y_pred)

        return scalars, projections


ALL_ANALYSES = {
    "pca": PCAAnalysis,
    "linear_regression": LinearRegressionAnalysis,
}
