"""Persistence helpers for activation visualization payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from simplexity.activations.activation_visualizations import (
    ActivationVisualizationPayload,
    render_visualization,
)
from simplexity.visualization.history import (
    history_paths,
    load_history_dataframe,
    plot_config_signature,
    save_history_dataframe,
)


def save_visualization_payloads(
    visualizations: Mapping[str, ActivationVisualizationPayload],
    root: Path,
    step: int,
) -> Mapping[str, str]:
    """Persist visualization payloads, accumulating history for slider controls."""

    if not visualizations:
        return {}

    figure_names_to_paths = {}
    step_dir = root / f"step_{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    for key, payload in visualizations.items():
        safe_name = key.replace("/", "_")
        figure = _maybe_accumulate_history(payload, root, safe_name, step)
        analysis_dir = step_dir / payload.analysis
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / f"{payload.name}.html"
        if payload.backend == "plotly":
            figure.write_html(str(output_path))
        else:
            figure.save(str(output_path), format="html")

        figure_names_to_paths[key] = str(output_path)

    return figure_names_to_paths

def _maybe_accumulate_history(
    payload: ActivationVisualizationPayload,
    root: Path,
    safe_name: str,
    step: int,
):
    if not _should_accumulate_steps(payload):
        return payload.figure

    data_path, meta_path = history_paths(root, safe_name)
    signature = plot_config_signature(payload.plot_config)
    existing_df = load_history_dataframe(data_path, meta_path, expected_signature=signature)
    new_rows = payload.dataframe.copy(deep=True)
    if "step" in new_rows.columns:
        new_rows["sequence_step"] = new_rows["step"]
    new_rows["step"] = step
    combined_df = pd.concat([existing_df, new_rows], ignore_index=True) if not existing_df.empty else new_rows
    slider = payload.controls.slider if payload.controls else None
    if slider and slider.field in combined_df.columns:
        slider.options = list(pd.unique(combined_df[slider.field]))
    save_history_dataframe(
        combined_df,
        data_path,
        meta_path,
        signature=signature,
        analysis=payload.analysis,
        name=payload.name,
        backend=payload.backend,
    )
    return render_visualization(payload.plot_config, combined_df, payload.controls)


def _should_accumulate_steps(payload: ActivationVisualizationPayload) -> bool:
    if payload.controls is None:
        return False
    if getattr(payload.controls, "accumulate_steps", False):
        return True
    slider = payload.controls.slider
    return slider is not None and slider.field == "step"


__all__ = ["save_visualization_payloads"]
