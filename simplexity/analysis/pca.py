import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from typing import Any, Dict, Optional, Sequence

from simplexity.utils.analysis_utils import build_prefix_dataset


def to_numpy(x):
    """Convert jax.Array or numpy array to numpy array."""
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return np.asarray(x)


def compute_pca(
    X,
    n_components: Optional[int] = None,
    weights=None,
    center: bool = True,
) -> Dict[str, Any]:
    """
    Compute PCA (optionally weighted) via eigen-decomposition of the covariance matrix.

    Args:
        X: (N, D) data matrix (torch.Tensor or np.ndarray)
        n_components: number of components to keep (default: all)
        weights: optional (N,) array / tensor of non-negative weights.
                 If given, they will be normalized to sum to 1.
        center: whether to subtract mean (True is standard PCA)

    Returns:
        dict with keys:
            - 'components': (n_components, D) principal axes (eigenvectors)
            - 'explained_variance': (n_components,) eigenvalues
            - 'explained_variance_ratio': (n_components,)
            - 'mean': (D,) mean vector used for centering
            - 'X_proj': (N, n_components) projected data
    """
    X = to_numpy(X)  # (N, D)
    N, D = X.shape

    if n_components is None:
        n_components = D
    n_components = min(n_components, D)

    # Handle weights
    if weights is None:
        w = None
        mean = X.mean(axis=0) if center else np.zeros(D, dtype=X.dtype)
    else:
        w = to_numpy(weights).astype(float)
        if w.ndim != 1 or w.shape[0] != N:
            raise ValueError(f"Weights must be shape (N,), got {w.shape} for N={N}")
        total = w.sum()
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        w = w / total  # normalize to sum 1

        if center:
            mean = np.average(X, axis=0, weights=w)
        else:
            mean = np.zeros(D, dtype=X.dtype)

    # Center data
    Xc = X - mean

    # Compute weighted covariance
    if w is None:
        # Unweighted covariance (population version)
        cov = (Xc.T @ Xc) / Xc.shape[0]
    else:
        # Weighted covariance: sum_i w_i (x_i - mu)(x_i - mu)^T
        # Implement as (Xc * w[:, None]).T @ Xc (since sum w_i = 1)
        cov = (Xc * w[:, None]).T @ Xc

    # Eigen decomposition; covariance is symmetric
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top components
    eigvals_sel = eigvals[:n_components]
    eigvecs_sel = eigvecs[:, :n_components]  # (D, n_components)

    total_var = eigvals.sum()
    if total_var <= 0:
        explained_ratio = np.zeros_like(eigvals_sel)
    else:
        explained_ratio = eigvals_sel / total_var

    # Project data
    X_proj = Xc @ eigvecs_sel  # (N, n_components)

    return {
        "components": eigvecs_sel.T,          # (n_components, D)
        "explained_variance": eigvals_sel,    # (n_components,)
        "explained_variance_ratio": explained_ratio,  # (n_components,)
        "mean": mean,                         # (D,)
        "X_proj": X_proj,                     # (N, n_components)
        "all_explained_variance": eigvals,    # (D,)
        "all_explained_variance_ratio": eigvals / total_var if total_var > 0 else np.zeros_like(eigvals),
    }

def plot_pca_2d(
    pca_res: Dict[str, Any],
    labels: Optional[Sequence] = None,
    title: str = "2D PCA Projection",
    marker_size: int = 6,
) -> go.Figure:
    """
    Compute PCA (optionally weighted) and plot the first two components with Plotly.

    Args:
        X: (N, D) data (torch.Tensor or np.ndarray)
        weights: optional (N,) weights (torch or np)
        labels: optional (N,) labels for coloring points
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure.
    """
    proj = pca_res["X_proj"]  # (N, 2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=proj[:, 0],
            y=proj[:, 1],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=labels if labels is not None else None,
                colorscale="Viridis" if labels is not None else None,
                showscale=labels is not None,
            ),
            text=labels,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
    )

    return fig

def plot_pca_3d(
    pca_res: Dict[str, Any],
    labels: Optional[Sequence] = None,
    title: str = "3D PCA Projection",
    marker_size: int = 4,
) -> go.Figure:
    """
    Compute PCA (optionally weighted) and plot the first three components with Plotly.

    Args:
        X: (N, D) data (torch.Tensor or np.ndarray)
        weights: optional (N,) weights
        labels: optional (N,) labels for coloring
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure.
    """
    proj = pca_res["X_proj"]  # (N, 3)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=labels if labels is not None else None,
                colorscale="Viridis" if labels is not None else None,
                showscale=labels is not None,
            ),
            text=labels,
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        template="plotly_white",
    )

    return fig

def plot_cumulative_explained_variance(
    pca_res: Dict[str, Any],
    title: str = "Cumulative Explained Variance",
) -> go.Figure:
    """
    Plot cumulative explained variance from PCA (optionally weighted).

    Args:
        X: (N, D) data (torch.Tensor or np.ndarray)
        weights: optional (N,) weights
        title: plot title

    Returns:
        Plotly Figure.
    """
    var_ratio_all = pca_res["all_explained_variance_ratio"]  # (D,)
    cum_var = np.cumsum(var_ratio_all)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(cum_var) + 1),
            y=cum_var,
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Explained Variance",
        template="plotly_white",
        yaxis=dict(range=[0, 1.01]),
    )

    return fig


def generate_pca_plots(
    X,
    n_components: Optional[int] = None,
    weights: Optional[jax.Array] = None,
    plot_2d: bool = True,
    plot_3d: bool = True,
    plot_cumulative_variance: bool = True,
) -> Dict[str, go.Figure]:
    """
    Compute PCA (optionally weighted) and generate standard PCA plots.

    Args:
        X: (N, D) data (jax.Array or np.ndarray)
        n_components: number of components to keep (default: all)
        weights: optional (N,) weights
        plot_2d: whether to generate 2D PCA plot
        plot_3d: whether to generate 3D PCA plot
        plot_cumulative_variance: whether to generate cumulative explained variance plot
    Returns:
        dict of plot name -> Plotly Figure
    """
    if not (plot_2d or plot_3d or plot_cumulative_variance):
        return {}

    pca_res = compute_pca(X, n_components=n_components, weights=weights)

    plots: Dict[str, go.Figure] = {}

    if plot_2d:
        fig_2d = plot_pca_2d(pca_res, title="2D PCA Projection")
        plots["pca_2d"] = fig_2d

    if plot_3d:
        fig_3d = plot_pca_3d(pca_res, title="3D PCA Projection")
        plots["pca_3d"] = fig_3d

    if plot_cumulative_variance:
        fig_cumvar = plot_cumulative_explained_variance(pca_res, title="Cumulative Explained Variance")
        plots["cumulative_explained_variance"] = fig_cumvar

    return plots


def compute_variance_thresholds(
    pca_res: Dict[str, Any],
    thresholds: Sequence[float],
) -> Dict[float, int]:
    """
    Compute the number of principal components needed to reach each variance threshold.

    Args:
        pca_res: PCA result dict from compute_pca
        thresholds: list of variance thresholds (e.g., [0.80, 0.90, 0.95, 0.99])

    Returns:
        dict mapping threshold -> number of components needed
    """
    cum_var = np.cumsum(pca_res["all_explained_variance_ratio"])
    result = {}

    for threshold in thresholds:
        # Find first index where cumulative variance >= threshold
        indices = np.where(cum_var >= threshold)[0]
        if len(indices) > 0:
            # +1 because we want number of components (1-indexed), not index (0-indexed)
            result[threshold] = int(indices[0]) + 1
        else:
            # If threshold is never reached, return total number of components
            result[threshold] = len(cum_var)

    return result


def pca_prefix_activations(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations_by_layer: Dict[str, jax.Array],
    layer_name: str,
    n_components: Optional[int] = None,
    plot_2d: bool = True,
    plot_3d: bool = True,
    plot_cumulative_variance: bool = True,
) -> Dict[str, go.Figure]:
    """
    Compute weighted PCA on prefix-deduplicated activations and generate plots.

    Weights each prefix by its summed probability (frequency) in the data.

    Args:
        inputs: (batch, seq_len) integer token ids
        beliefs: (batch, seq_len, B) beliefs at each position
        probs: (batch, seq_len) probabilities at each position
        activations_by_layer: layer -> (batch, seq_len, d_layer) activations
        layer_name: which layer's activations to analyze
        n_components: number of PCA components to compute (default: all)
        plot_2d: whether to generate 2D PCA plot
        plot_3d: whether to generate 3D PCA plot
        plot_cumulative_variance: whether to generate cumulative explained variance plot

    Returns:
        dict of plot name -> Plotly Figure
    """
    prefix_dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)
    X = prefix_dataset.activations_by_layer[layer_name]
    weights = prefix_dataset.probs

    return generate_pca_plots(
        X,
        n_components=n_components,
        weights=weights,
        plot_2d=plot_2d,
        plot_3d=plot_3d,
        plot_cumulative_variance=plot_cumulative_variance,
    )


def plot_pca_2d_with_step_slider(
    pca_results_by_step: Dict[int, Dict[str, Any]],
    labels_by_step: Optional[Dict[int, Sequence]] = None,
    title: str = "2D PCA Projection Over Training Steps",
    marker_size: int = 6,
) -> go.Figure:
    """
    Create interactive 2D PCA plot with slider to navigate through training steps.

    Args:
        pca_results_by_step: dict mapping step -> PCA result from compute_pca()
        labels_by_step: optional dict mapping step -> labels for coloring
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with step slider
    """
    steps_sorted = sorted(pca_results_by_step.keys())

    if not steps_sorted:
        raise ValueError("pca_results_by_step must not be empty")

    fig = go.Figure()

    # Create traces for each step (initially all hidden)
    for step in steps_sorted:
        pca_res = pca_results_by_step[step]
        proj = pca_res["X_proj"][:, :2]  # first 2 components
        labels = labels_by_step.get(step) if labels_by_step else None

        fig.add_trace(
            go.Scatter(
                x=proj[:, 0],
                y=proj[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=labels if labels is not None else None,
                    colorscale="Viridis" if labels is not None else None,
                    showscale=labels is not None,
                ),
                text=labels,
                name=f"Step {step}",
                visible=False,
            )
        )

    # Make first trace visible
    fig.data[0].visible = True

    # Create slider steps
    slider_steps = []
    for i, step in enumerate(steps_sorted):
        step_dict = {
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(steps_sorted))]},
                {"title": f"{title} (Step {step})"},
            ],
            "label": str(step),
        }
        slider_steps.append(step_dict)

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"{title} (Step {steps_sorted[0]})",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_pca_2d_with_layer_dropdown(
    pca_results_by_layer: Dict[str, Dict[str, Any]],
    labels_by_layer: Optional[Dict[str, Sequence]] = None,
    title: str = "2D PCA Projection Across Layers",
    marker_size: int = 6,
) -> go.Figure:
    """
    Create interactive 2D PCA plot with dropdown to switch between layers.

    Args:
        pca_results_by_layer: dict mapping layer_name -> PCA result from compute_pca()
        labels_by_layer: optional dict mapping layer_name -> labels for coloring
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with layer dropdown
    """
    layers_sorted = sorted(pca_results_by_layer.keys())

    if not layers_sorted:
        raise ValueError("pca_results_by_layer must not be empty")

    fig = go.Figure()

    # Create traces for each layer (initially all hidden)
    for layer_name in layers_sorted:
        pca_res = pca_results_by_layer[layer_name]
        proj = pca_res["X_proj"][:, :2]
        labels = labels_by_layer.get(layer_name) if labels_by_layer else None

        fig.add_trace(
            go.Scatter(
                x=proj[:, 0],
                y=proj[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=labels if labels is not None else None,
                    colorscale="Viridis" if labels is not None else None,
                    showscale=labels is not None,
                ),
                text=labels,
                name=layer_name,
                visible=False,
            )
        )

    # Make first trace visible
    fig.data[0].visible = True

    # Create dropdown buttons
    buttons = []
    for i, layer_name in enumerate(layers_sorted):
        button_dict = {
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(layers_sorted))]},
                {"title": f"{title} ({layer_name})"},
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} ({layers_sorted[0]})",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
        height=600,
    )

    return fig


def plot_pca_2d_with_step_and_layer(
    pca_results_by_step_and_layer: Dict[int, Dict[str, Dict[str, Any]]],
    labels_by_step_and_layer: Optional[Dict[int, Dict[str, Sequence]]] = None,
    title: str = "2D PCA Projection",
    marker_size: int = 6,
) -> go.Figure:
    """
    Create interactive 2D PCA plot with both step slider and layer dropdown.

    Args:
        pca_results_by_step_and_layer: dict mapping step -> layer_name -> PCA result
        labels_by_step_and_layer: optional dict mapping step -> layer_name -> labels
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with step slider and layer dropdown
    """
    steps_sorted = sorted(pca_results_by_step_and_layer.keys())
    if not steps_sorted:
        raise ValueError("pca_results_by_step_and_layer must not be empty")

    # Get all layer names (assume same layers at each step)
    layers_sorted = sorted(pca_results_by_step_and_layer[steps_sorted[0]].keys())

    fig = go.Figure()

    # Create traces for each (step, layer) combination
    for step in steps_sorted:
        for layer_name in layers_sorted:
            pca_res = pca_results_by_step_and_layer[step][layer_name]
            proj = pca_res["X_proj"][:, :2]
            labels = None
            if labels_by_step_and_layer and step in labels_by_step_and_layer:
                labels = labels_by_step_and_layer[step].get(layer_name)

            fig.add_trace(
                go.Scatter(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=labels if labels is not None else None,
                        colorscale="Viridis" if labels is not None else None,
                        showscale=labels is not None,
                    ),
                    text=labels,
                    name=f"Step {step}, {layer_name}",
                    visible=False,
                )
            )

    # Make first trace visible
    fig.data[0].visible = True

    # Create a single slider that updates visibility based on step
    # Dropdown will update the slider configuration to work with the selected layer

    # Build slider steps for first layer
    def make_slider_for_layer(layer_idx: int, layer_name: str):
        slider_steps = []
        for step_idx, step in enumerate(steps_sorted):
            # Show only traces for this step and this layer
            visible_flags = []
            for s_idx in range(len(steps_sorted)):
                for l_idx in range(len(layers_sorted)):
                    visible_flags.append(s_idx == step_idx and l_idx == layer_idx)

            slider_steps.append({
                "method": "update",
                "args": [
                    {"visible": visible_flags},
                    {"title": f"{title} (Step {step}, {layer_name})"},
                ],
                "label": str(step),
            })

        return {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }

    # Create dropdown that switches between layer-specific sliders
    buttons = []
    sliders_by_layer = []

    for layer_idx, layer_name in enumerate(layers_sorted):
        # Create slider for this layer
        layer_slider = make_slider_for_layer(layer_idx, layer_name)
        sliders_by_layer.append(layer_slider)

        # Dropdown button switches to this layer's slider and shows first step
        visible_flags = []
        for s_idx in range(len(steps_sorted)):
            for l_idx in range(len(layers_sorted)):
                visible_flags.append(s_idx == 0 and l_idx == layer_idx)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {
                    "sliders": [layer_slider],
                    "title": f"{title} (Step {steps_sorted[0]}, {layer_name})",
                },
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    # Start with the first layer's slider
    fig.update_layout(
        sliders=[sliders_by_layer[0]],
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=title,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_white",
        height=600,
    )

    return fig


# ===========================
# 3D PCA Plotting Functions
# ===========================


def plot_pca_3d_with_step_slider(
    pca_results_by_step: Dict[int, Dict[str, Any]],
    labels_by_step: Optional[Dict[int, Sequence]] = None,
    title: str = "3D PCA Projection Over Training Steps",
    marker_size: int = 4,
) -> go.Figure:
    """
    Create interactive 3D PCA plot with slider to navigate through training steps.

    Args:
        pca_results_by_step: dict mapping step -> PCA result from compute_pca()
        labels_by_step: optional dict mapping step -> labels for coloring
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with step slider
    """
    steps_sorted = sorted(pca_results_by_step.keys())

    if not steps_sorted:
        raise ValueError("pca_results_by_step must not be empty")

    fig = go.Figure()

    # Create traces for each step (initially all hidden)
    for step in steps_sorted:
        pca_res = pca_results_by_step[step]
        proj = pca_res["X_proj"][:, :3]  # first 3 components
        labels = labels_by_step.get(step) if labels_by_step else None

        fig.add_trace(
            go.Scatter3d(
                x=proj[:, 0],
                y=proj[:, 1],
                z=proj[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=labels if labels is not None else None,
                    colorscale="Viridis" if labels is not None else None,
                    showscale=labels is not None,
                ),
                text=labels,
                name=f"Step {step}",
                visible=False,
            )
        )

    # Make first trace visible
    fig.data[0].visible = True

    # Create slider steps
    slider_steps = []
    for i, step in enumerate(steps_sorted):
        step_dict = {
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(steps_sorted))]},
                {"title": f"{title} (Step {step})"},
            ],
            "label": str(step),
        }
        slider_steps.append(step_dict)

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"{title} (Step {steps_sorted[0]})",
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        template="plotly_white",
        height=700,
    )

    return fig


def plot_pca_3d_with_layer_dropdown(
    pca_results_by_layer: Dict[str, Dict[str, Any]],
    labels_by_layer: Optional[Dict[str, Sequence]] = None,
    title: str = "3D PCA Projection Across Layers",
    marker_size: int = 4,
) -> go.Figure:
    """
    Create interactive 3D PCA plot with dropdown to switch between layers.

    Args:
        pca_results_by_layer: dict mapping layer_name -> PCA result from compute_pca()
        labels_by_layer: optional dict mapping layer_name -> labels for coloring
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with layer dropdown
    """
    layers_sorted = sorted(pca_results_by_layer.keys())

    if not layers_sorted:
        raise ValueError("pca_results_by_layer must not be empty")

    fig = go.Figure()

    # Create traces for each layer (initially all hidden)
    for layer_name in layers_sorted:
        pca_res = pca_results_by_layer[layer_name]
        proj = pca_res["X_proj"][:, :3]  # first 3 components
        labels = labels_by_layer.get(layer_name) if labels_by_layer else None

        fig.add_trace(
            go.Scatter3d(
                x=proj[:, 0],
                y=proj[:, 1],
                z=proj[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=labels if labels is not None else None,
                    colorscale="Viridis" if labels is not None else None,
                    showscale=labels is not None,
                ),
                text=labels,
                name=layer_name,
                visible=False,
            )
        )

    # Make first trace visible
    fig.data[0].visible = True

    # Create dropdown menu
    dropdown_buttons = []
    for i, layer_name in enumerate(layers_sorted):
        button_dict = {
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(layers_sorted))]},
                {"title": f"{title} ({layer_name})"},
            ],
            "label": layer_name,
        }
        dropdown_buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.15,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} ({layers_sorted[0]})",
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        template="plotly_white",
        height=700,
    )

    return fig


def plot_pca_3d_with_step_and_layer(
    pca_results_by_step_and_layer: Dict[int, Dict[str, Dict[str, Any]]],
    labels_by_step_and_layer: Optional[Dict[int, Dict[str, Sequence]]] = None,
    title: str = "3D PCA: Training Evolution Across Layers",
    marker_size: int = 4,
) -> go.Figure:
    """
    Create interactive 3D PCA plot with both step slider and layer dropdown.

    Args:
        pca_results_by_step_and_layer: nested dict {step: {layer_name: pca_result}}
        labels_by_step_and_layer: optional nested dict {step: {layer_name: labels}}
        title: plot title
        marker_size: point size

    Returns:
        Plotly Figure with step slider and layer dropdown
    """
    steps_sorted = sorted(pca_results_by_step_and_layer.keys())
    if not steps_sorted:
        raise ValueError("pca_results_by_step_and_layer must not be empty")

    # Get all unique layers across all steps
    all_layers = set()
    for step_results in pca_results_by_step_and_layer.values():
        all_layers.update(step_results.keys())
    layers_sorted = sorted(all_layers)

    fig = go.Figure()

    # Create traces for each (step, layer) combination
    for step in steps_sorted:
        for layer_name in layers_sorted:
            if layer_name not in pca_results_by_step_and_layer[step]:
                continue

            pca_res = pca_results_by_step_and_layer[step][layer_name]
            proj = pca_res["X_proj"][:, :3]  # first 3 components

            labels = None
            if labels_by_step_and_layer and step in labels_by_step_and_layer:
                labels = labels_by_step_and_layer[step].get(layer_name)

            fig.add_trace(
                go.Scatter3d(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    z=proj[:, 2],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=labels if labels is not None else None,
                        colorscale="Viridis" if labels is not None else None,
                        showscale=labels is not None,
                    ),
                    text=labels,
                    name=f"Step {step}, {layer_name}",
                    visible=False,
                )
            )

    # Make first trace visible
    fig.data[0].visible = True

    # Build slider for a specific layer
    def make_slider_for_layer_3d(layer_idx: int, layer_name: str):
        slider_steps = []
        for step_idx, step in enumerate(steps_sorted):
            # Show only traces for this step and this layer
            visible_flags = []
            trace_idx = 0
            for s_idx, s in enumerate(steps_sorted):
                for l_idx, layer in enumerate(layers_sorted):
                    if layer not in pca_results_by_step_and_layer[s]:
                        continue
                    visible_flags.append(s_idx == step_idx and l_idx == layer_idx)
                    trace_idx += 1

            slider_steps.append({
                "method": "update",
                "args": [
                    {"visible": visible_flags},
                    {"title": f"{title} (Step {step}, {layer_name})"},
                ],
                "label": str(step),
            })

        return {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }

    # Create dropdown that switches between layer-specific sliders
    dropdown_buttons = []
    sliders_by_layer = []

    for layer_idx, layer_name in enumerate(layers_sorted):
        # Create slider for this layer
        layer_slider = make_slider_for_layer_3d(layer_idx, layer_name)
        sliders_by_layer.append(layer_slider)

        # Dropdown button switches to this layer's slider
        visible_flags = []
        for s_idx, s in enumerate(steps_sorted):
            for l_idx, layer in enumerate(layers_sorted):
                if layer not in pca_results_by_step_and_layer[s]:
                    continue
                visible_flags.append(s_idx == 0 and l_idx == layer_idx)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {
                    "sliders": [layer_slider],
                    "title": f"{title} (Step {steps_sorted[0]}, {layer_name})",
                },
            ],
            "label": layer_name,
        }
        dropdown_buttons.append(button_dict)

    # Start with the first layer's slider
    fig.update_layout(
        sliders=[sliders_by_layer[0]],
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.15,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=title,
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
        ),
        template="plotly_white",
        height=700,
    )

    return fig


# ===========================
# Variance Explained Plotting
# ===========================


def plot_variance_explained(
    pca_results_by_step: Optional[Dict[int, Dict[str, Any]]] = None,
    pca_results_by_layer: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "PCA Variance Explained",
    max_components: Optional[int] = None,
) -> go.Figure:
    """
    Create scree plot showing variance explained by each principal component.

    Args:
        pca_results_by_step: optional dict mapping step -> PCA result
        pca_results_by_layer: optional dict mapping layer_name -> PCA result
        title: plot title
        max_components: maximum number of components to display

    Returns:
        Plotly Figure showing variance explained (scree plot)
    """
    if pca_results_by_step is None and pca_results_by_layer is None:
        raise ValueError("Must provide either pca_results_by_step or pca_results_by_layer")

    fig = go.Figure()

    # Plot by step
    if pca_results_by_step is not None:
        for step in sorted(pca_results_by_step.keys()):
            pca_res = pca_results_by_step[step]
            var_ratio = pca_res["explained_variance_ratio"]
            if max_components is not None:
                var_ratio = var_ratio[:max_components]

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(var_ratio) + 1)),
                    y=var_ratio,
                    mode="lines+markers",
                    name=f"Step {step}",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    # Plot by layer
    if pca_results_by_layer is not None:
        for layer_name in sorted(pca_results_by_layer.keys()):
            pca_res = pca_results_by_layer[layer_name]
            var_ratio = pca_res["explained_variance_ratio"]
            if max_components is not None:
                var_ratio = var_ratio[:max_components]

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(var_ratio) + 1)),
                    y=var_ratio,
                    mode="lines+markers",
                    name=layer_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained Ratio",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_cumulative_variance_explained(
    pca_results_by_step: Optional[Dict[int, Dict[str, Any]]] = None,
    pca_results_by_layer: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "Cumulative Variance Explained",
    max_components: Optional[int] = None,
    thresholds: Optional[list[float]] = None,
) -> go.Figure:
    """
    Create plot showing cumulative variance explained by principal components.

    The plot shows cumulative variance (y-axis) vs number of components (x-axis).
    Horizontal dashed lines mark the variance threshold levels (e.g., 90%, 95%, 99%).

    Args:
        pca_results_by_step: optional dict mapping step -> PCA result
        pca_results_by_layer: optional dict mapping layer_name -> PCA result
        title: plot title
        max_components: maximum number of components to display
        thresholds: optional list of variance thresholds to mark with horizontal lines
                   (e.g., [0.8, 0.9, 0.95])

    Returns:
        Plotly Figure showing cumulative variance explained with horizontal threshold lines
    """
    if pca_results_by_step is None and pca_results_by_layer is None:
        raise ValueError("Must provide either pca_results_by_step or pca_results_by_layer")

    fig = go.Figure()

    # Track all cumsum curves and their threshold crossings
    all_cumsums = []
    all_names = []

    # Plot by step
    if pca_results_by_step is not None:
        for step in sorted(pca_results_by_step.keys()):
            pca_res = pca_results_by_step[step]
            var_ratio = pca_res["explained_variance_ratio"]
            cumsum = np.cumsum(var_ratio)
            if max_components is not None:
                cumsum = cumsum[:max_components]

            all_cumsums.append(cumsum)
            all_names.append(f"Step {step}")

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum) + 1)),
                    y=cumsum,
                    mode="lines+markers",
                    name=f"Step {step}",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    # Plot by layer
    if pca_results_by_layer is not None:
        for layer_name in sorted(pca_results_by_layer.keys()):
            pca_res = pca_results_by_layer[layer_name]
            var_ratio = pca_res["explained_variance_ratio"]
            cumsum = np.cumsum(var_ratio)
            if max_components is not None:
                cumsum = cumsum[:max_components]

            all_cumsums.append(cumsum)
            all_names.append(layer_name)

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum) + 1)),
                    y=cumsum,
                    mode="lines+markers",
                    name=layer_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    # Add horizontal lines at variance threshold levels
    if thresholds is not None:
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"{threshold:.0%}",
                annotation_position="right",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_cumulative_variance_with_step_dropdown(
    pca_results_by_step_and_layer: Dict[int, Dict[str, Dict[str, Any]]],
    title: str = "Cumulative Variance Explained by Step",
    max_components: Optional[int] = None,
    thresholds: Optional[list[float]] = None,
) -> go.Figure:
    """
    Create interactive cumulative variance plot with dropdown to select step.
    Shows all layers at the selected step.

    Args:
        pca_results_by_step_and_layer: nested dict {step: {layer_name: pca_result}}
        title: plot title
        max_components: maximum number of components to display
        thresholds: optional list of variance thresholds to mark with horizontal lines

    Returns:
        Plotly Figure with step dropdown
    """
    steps_sorted = sorted(pca_results_by_step_and_layer.keys())
    if not steps_sorted:
        raise ValueError("pca_results_by_step_and_layer must not be empty")

    fig = go.Figure()

    # Create traces for each (step, layer) combination
    for step in steps_sorted:
        for layer_name in sorted(pca_results_by_step_and_layer[step].keys()):
            pca_res = pca_results_by_step_and_layer[step][layer_name]
            var_ratio = pca_res["explained_variance_ratio"]
            cumsum = np.cumsum(var_ratio)
            if max_components is not None:
                cumsum = cumsum[:max_components]

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum) + 1)),
                    y=cumsum,
                    mode="lines+markers",
                    name=layer_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                    visible=False,
                )
            )

    # Make first step's traces visible
    layers_per_step = len(pca_results_by_step_and_layer[steps_sorted[0]])
    for i in range(layers_per_step):
        fig.data[i].visible = True

    # Add horizontal lines at variance threshold levels
    if thresholds is not None:
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"{threshold:.0%}",
                annotation_position="right",
            )

    # Create dropdown buttons for steps
    buttons = []
    trace_idx = 0
    for step_idx, step in enumerate(steps_sorted):
        layers_at_step = sorted(pca_results_by_step_and_layer[step].keys())
        n_layers = len(layers_at_step)

        # Visibility: show only traces for this step
        visible_flags = []
        for s_idx in range(len(steps_sorted)):
            step_layers = sorted(pca_results_by_step_and_layer[steps_sorted[s_idx]].keys())
            for _ in step_layers:
                visible_flags.append(s_idx == step_idx)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} (Step {step})"},
            ],
            "label": f"Step {step}",
        }
        buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} (Step {steps_sorted[0]})",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_cumulative_variance_with_layer_dropdown(
    pca_results_by_step_and_layer: Dict[int, Dict[str, Dict[str, Any]]],
    title: str = "Cumulative Variance Explained by Layer",
    max_components: Optional[int] = None,
    thresholds: Optional[list[float]] = None,
) -> go.Figure:
    """
    Create interactive cumulative variance plot with dropdown to select layer.
    Shows all steps for the selected layer.

    Args:
        pca_results_by_step_and_layer: nested dict {step: {layer_name: pca_result}}
        title: plot title
        max_components: maximum number of components to display
        thresholds: optional list of variance thresholds to mark with horizontal lines

    Returns:
        Plotly Figure with layer dropdown
    """
    steps_sorted = sorted(pca_results_by_step_and_layer.keys())
    if not steps_sorted:
        raise ValueError("pca_results_by_step_and_layer must not be empty")

    # Get all unique layers
    all_layers = set()
    for step_data in pca_results_by_step_and_layer.values():
        all_layers.update(step_data.keys())
    layers_sorted = sorted(all_layers)

    fig = go.Figure()

    # Create traces for each (layer, step) combination (note: swapped order from above)
    for layer_name in layers_sorted:
        for step in steps_sorted:
            if layer_name not in pca_results_by_step_and_layer[step]:
                continue

            pca_res = pca_results_by_step_and_layer[step][layer_name]
            var_ratio = pca_res["explained_variance_ratio"]
            cumsum = np.cumsum(var_ratio)
            if max_components is not None:
                cumsum = cumsum[:max_components]

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum) + 1)),
                    y=cumsum,
                    mode="lines+markers",
                    name=f"Step {step}",
                    line=dict(width=2),
                    marker=dict(size=6),
                    visible=False,
                )
            )

    # Make first layer's traces visible
    first_layer_trace_count = sum(
        1 for step in steps_sorted
        if layers_sorted[0] in pca_results_by_step_and_layer[step]
    )
    for i in range(first_layer_trace_count):
        fig.data[i].visible = True

    # Add horizontal lines at variance threshold levels
    if thresholds is not None:
        for threshold in thresholds:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"{threshold:.0%}",
                annotation_position="right",
            )

    # Create dropdown buttons for layers
    buttons = []
    for layer_idx, layer_name in enumerate(layers_sorted):
        # Count traces for each layer
        visible_flags = []
        for l_idx in range(len(layers_sorted)):
            for step in steps_sorted:
                if layers_sorted[l_idx] not in pca_results_by_step_and_layer[step]:
                    continue
                visible_flags.append(l_idx == layer_idx)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} ({layer_name})"},
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} ({layers_sorted[0]})",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_components_for_variance_threshold(
    variance_threshold_results: Dict[int, Dict[str, Dict[float, int]]],
    title: str = "Components Required for Variance Thresholds",
) -> go.Figure:
    """
    Plot number of components required to reach variance thresholds over training.

    Args:
        variance_threshold_results: nested dict {step: {layer_name: {threshold: n_components}}}
        title: plot title

    Returns:
        Plotly Figure showing components vs. training step
    """
    if not variance_threshold_results:
        raise ValueError("variance_threshold_results must not be empty")

    steps_sorted = sorted(variance_threshold_results.keys())

    # Get all layers and thresholds
    all_layers = set()
    all_thresholds = set()
    for step_results in variance_threshold_results.values():
        all_layers.update(step_results.keys())
        for threshold_dict in step_results.values():
            all_thresholds.update(threshold_dict.keys())

    layers_sorted = sorted(all_layers)
    thresholds_sorted = sorted(all_thresholds)

    fig = go.Figure()

    # Create traces for each (layer, threshold) combination
    for layer_name in layers_sorted:
        for threshold in thresholds_sorted:
            n_components_over_steps = []
            for step in steps_sorted:
                if layer_name in variance_threshold_results[step]:
                    n_comp = variance_threshold_results[step][layer_name].get(threshold)
                    n_components_over_steps.append(n_comp)
                else:
                    n_components_over_steps.append(None)

            fig.add_trace(
                go.Scatter(
                    x=steps_sorted,
                    y=n_components_over_steps,
                    mode="lines+markers",
                    name=f"{layer_name} ({threshold:.0%})",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Training Step",
        yaxis_title="Number of Components",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig