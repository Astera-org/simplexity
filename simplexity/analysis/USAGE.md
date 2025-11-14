# Analysis Tracker Usage Guide

This guide shows how to integrate analysis tracking into your training/evaluation loops.

## Quick Start

```python
from simplexity.analysis import AnalysisTracker

# Option 1: Auto-detect all layers (recommended)
tracker = AnalysisTracker()

# Option 2: Explicitly specify layer names
tracker = AnalysisTracker(
    layer_names=["layer_0", "layer_1", "layer_2", "output"],
    variance_thresholds=[0.80, 0.90, 0.95, 0.99]
)

# During validation loops
for step in validation_steps:
    # Get your data (however you compute it)
    inputs, beliefs, probs, activations = get_validation_data(model, step)

    # Add to tracker - layers are auto-detected from first call
    tracker.add_step(
        step=step,
        inputs=inputs,
        beliefs=beliefs,
        probs=probs,
        activations_by_layer=activations,
    )

# Generate all interactive plots
tracker.save_all_plots(output_dir="analysis_results/")
```

## Integration with Training Loop

```python
from simplexity.analysis import AnalysisTracker

def train_model(model, train_data, val_data, config):
    # Initialize tracker - layers will be auto-detected
    tracker = AnalysisTracker(variance_thresholds=[0.80, 0.90, 0.95, 0.99])

    for epoch in range(config.n_epochs):
        # Training
        for batch in train_data:
            loss = train_step(model, batch)

        # Validation
        if epoch % config.val_every_n_epochs == 0:
            # Run validation and collect activations
            val_results = validate(model, val_data)

            # Add to tracker
            tracker.add_step(
                step=epoch,
                inputs=val_results["inputs"],
                beliefs=val_results["beliefs"],
                probs=val_results["probs"],
                activations_by_layer=val_results["activations"],
            )

            # Optional: print summaries
            print_analysis_summary(tracker, epoch)

    # After training: generate and save plots
    tracker.save_all_plots(f"results/run_{config.run_id}/")

    # Optional: also get metric summaries
    regression_summary = tracker.get_regression_metrics_summary()
    variance_summary = tracker.get_variance_threshold_summary()

    return model, tracker, regression_summary, variance_summary
```

## Selective Analysis

You can control which analyses to run:

```python
# Only PCA, skip regression (faster)
tracker.add_step(
    step=step,
    inputs=inputs,
    beliefs=beliefs,
    probs=probs,
    activations_by_layer=activations,
    compute_pca=True,
    compute_regression=False,
)

# Only regression, skip PCA
tracker.add_step(
    step=step,
    inputs=inputs,
    beliefs=beliefs,
    probs=probs,
    activations_by_layer=activations,
    compute_pca=False,
    compute_regression=True,
)
```

## Accessing Results Programmatically

```python
# Get specific results
pca_result = tracker.get_pca_result(step=1000, layer_name="layer_2")
regression_result = tracker.get_regression_result(step=1000, layer_name="layer_2")

# Check R² score
print(f"R² at step 1000, layer 2: {regression_result.r2:.3f}")

# Check how many components needed for 90% variance
variance_thresholds = tracker.get_variance_thresholds(step=1000, layer_name="layer_2")
print(f"Components for 90% variance: {variance_thresholds[0.90]}")
```

## Getting Metric Summaries

```python
# Regression metrics over time
regression_summary = tracker.get_regression_metrics_summary()

# By layer: see how each layer evolves
for layer_name, metrics in regression_summary["by_layer"].items():
    r2_over_time = metrics["r2"]
    print(f"{layer_name}: R² progression: {r2_over_time}")

# By step: compare layers at each checkpoint
for step, layer_metrics in regression_summary["by_step"].items():
    print(f"\nStep {step}:")
    for layer_name, metrics in layer_metrics.items():
        print(f"  {layer_name}: R²={metrics['r2']:.3f}, dist={metrics['dist']:.3f}")
```

## Variance Threshold Analysis

```python
variance_summary = tracker.get_variance_threshold_summary()

# See component requirements over training
for layer_name, thresholds in variance_summary["by_layer"].items():
    components_90 = thresholds[0.90]  # list of n_components across steps
    print(f"{layer_name}: Components for 90% variance over time: {components_90}")
```

## Generating Specific Plot Types

```python
# Only generate combined plots (most comprehensive)
plots = tracker.generate_pca_plots(by_step=False, by_layer=False, combined=True)
plots["pca_combined"].show()

# Only generate step sliders for each layer
plots = tracker.generate_pca_plots(by_step=True, by_layer=False, combined=False)
for name, fig in plots.items():
    fig.show()

# Generate regression plots only
regression_plots = tracker.generate_regression_plots()
regression_plots["regression_combined"].write_html("regression_analysis.html")
```

## Memory Management

For long training runs, you can clear the tracker periodically:

```python
# Track validation every 100 steps
for step in range(0, 100000, 100):
    tracker.add_step(...)

    # Save and clear every 1000 steps to manage memory
    if step % 1000 == 0:
        tracker.save_all_plots(f"checkpoints/step_{step}/")
        tracker.clear()
```

## Example: Full Training Script

```python
import jax
import equinox as eqx
from simplexity.analysis import AnalysisTracker

def extract_activations(model, inputs):
    """Extract layer activations from model."""
    # This is model-specific - adapt to your architecture
    activations = {}

    # Example for a sequential model
    x = inputs
    for i, layer in enumerate(model.layers):
        x = layer(x)
        activations[f"layer_{i}"] = x

    return activations

def main():
    # Setup
    model = build_model()
    tracker = AnalysisTracker(layer_names=[f"layer_{i}" for i in range(4)])

    # Training loop
    for step in range(10000):
        # Train
        model, loss = train_step(model, ...)

        # Validation analysis every 100 steps
        if step % 100 == 0:
            # Get validation batch
            val_batch = next(val_loader)

            # Extract activations
            activations = extract_activations(model, val_batch["inputs"])

            # Add to tracker
            tracker.add_step(
                step=step,
                inputs=val_batch["inputs"],
                beliefs=val_batch["beliefs"],
                probs=val_batch["probs"],
                activations_by_layer=activations,
            )

    # Final analysis
    tracker.save_all_plots("final_results/")

    # Print summary
    regression_summary = tracker.get_regression_metrics_summary()
    print("\nFinal R² by layer:")
    for layer_name, metrics in regression_summary["by_layer"].items():
        final_r2 = metrics["r2"][-1]
        print(f"  {layer_name}: {final_r2:.3f}")

if __name__ == "__main__":
    main()
```

## Plot Types Generated

The tracker generates several interactive HTML plots:

### PCA Plots
- `pca_step_slider_{layer}.html` - PCA for one layer across training steps
- `pca_layer_dropdown_step_{step}.html` - Compare layers at one checkpoint
- `pca_combined.html` - Both step slider and layer dropdown

### Regression Plots
- `regression_step_slider_{layer}.html` - Simplex projection for one layer across steps
- `regression_layer_dropdown_step_{step}.html` - Compare layers at one checkpoint
- `regression_combined.html` - Both step slider and layer dropdown

All plots are fully interactive - zoom, pan, hover for details, and use sliders/dropdowns to navigate.
