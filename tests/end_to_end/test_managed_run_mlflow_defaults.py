"""End-to-end tests for managed run with MLflow defaults."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import simplexity

CONFIG_DIR = str(Path(__file__).parent / "mlflow_defaults_configs")


def test_managed_run_loads_mlflow_defaults(setup_dir: Path) -> None:
    """Verify that managed_run correctly loads and merges MLflow default configs."""
    tracking_uri = f"sqlite:///{setup_dir.resolve()}/mlflow.db"
    with initialize_config_dir(config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="full_default_copy",
            overrides=[f"load_source.tracking_uri={tracking_uri}"],
        )

    captured: dict[str, DictConfig] = {}

    @simplexity.managed_run(strict=False)
    def run(cfg: DictConfig, components: simplexity.Components) -> None:  # pylint: disable=unused-argument
        captured["cfg"] = cfg

    run(cfg)  # pylint: disable=no-value-for-parameter

    actual = captured["cfg"]
    experiment_id = OmegaConf.select(actual, "load_source.experiment_id")
    run_id = OmegaConf.select(actual, "load_source.run_id")
    assert experiment_id is not None
    assert run_id is not None

    expected_overrides = [
        f"load_source.tracking_uri={tracking_uri}",
        f'load_source.experiment_id="{experiment_id}"',
        f'load_source.run_id="{run_id}"',
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR):
        expected = compose(config_name="full_default_copy_expected", overrides=expected_overrides)

    actual_container = OmegaConf.to_container(actual, resolve=True)
    actual_filtered = OmegaConf.create(actual_container)
    assert isinstance(actual_filtered, DictConfig)
    for key in ("device", "seed", "tags"):
        if key in actual_filtered:
            del actual_filtered[key]

    assert actual_filtered == expected
