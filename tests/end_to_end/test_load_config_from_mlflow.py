"""End-to-end mlflow defaults tests."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import mlflow
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from simplexity.structured_configs.mlflow_defaults import load_mlflow_defaults

CONFIG_DIR = str(Path(__file__).parent / "mlflow_defaults_configs")

EXPERIMENT_NAME = "test_mlflow_defaults"


def test_setup(setup_dir: Path, tmp_path: Path) -> None:
    """Test setup."""
    tracking_uri = f"sqlite:///{setup_dir.resolve()}/mlflow.db"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiments = client.get_experiment_by_name(name=EXPERIMENT_NAME)
    assert experiments is not None
    experiment_id = experiments.experiment_id

    for run_num in (1, 2):
        run = client.search_runs(
            experiment_ids=[experiment_id], filter_string=f"attributes.run_name = 'prev_run_{run_num}'"
        )
        assert len(run) == 1
        run_id = run[0].info.run_id

        config_path = client.download_artifacts(
            run_id=run_id, path="config.yaml", dst_path=str(tmp_path / f"config_{run_num}.yaml")
        )

        actual = OmegaConf.load(config_path)
        with initialize_config_dir(config_dir=str(setup_dir / "configs")):
            expected = compose(config_name=f"prev_config_{run_num}.yaml")
        assert actual == expected

        downloaded_dir = client.download_artifacts(
            run_id=run_id,
            path="subdir",
            dst_path=str(tmp_path),
        )
        special_config_path = Path(downloaded_dir) / f"special_{run_num}.yaml"
        expected_special_config_path = setup_dir / "configs" / f"special_{run_num}.yaml"
        with (
            open(special_config_path, encoding="utf-8") as f,
            open(expected_special_config_path, encoding="utf-8") as expected_f,
        ):
            assert f.read() == expected_f.read()


@pytest.mark.parametrize(
    ("test_case", "load_sources"),
    [
        ("full_default_copy", ["load_source"]),
        ("load_default_config_at_package", ["load_source"]),
        ("load_nondefault_config", ["load_source"]),
        ("load_subconfig", ["load_source"]),
        ("load_subconfig_select", ["load_source"]),
        ("implicit_config_select", ["load_source"]),
        ("implicit_artifact_select", ["load_source"]),
        ("override_flag", ["load_source"]),
        ("composition_order_self_first", ["load_source"]),
        ("composition_order_self_last", ["load_source"]),
        ("multiple_runs", ["load_source_1", "load_source_2"]),
    ],
)
def test_mlflow_defaults(setup_dir: Path, test_case: str, load_sources: list[str]) -> None:
    """Test mlflow defaults."""
    tracking_uri = f"sqlite:///{setup_dir.resolve()}/mlflow.db"
    with initialize_config_dir(config_dir=CONFIG_DIR):
        # Generate overrides for all load sources
        overrides = [f"{load_source}.tracking_uri={tracking_uri}" for load_source in load_sources]
        cfg = compose(config_name=test_case, overrides=overrides)
    actual = load_mlflow_defaults(cfg)

    # Extract experiment_id and run_id for all load sources
    expected_overrides = []
    for load_source in load_sources:
        experiment_id = OmegaConf.select(actual, f"{load_source}.experiment_id")
        run_id = OmegaConf.select(actual, f"{load_source}.run_id")
        assert experiment_id is not None
        assert run_id is not None
        expected_overrides.extend(
            [
                f"{load_source}.tracking_uri={tracking_uri}",
                f'{load_source}.experiment_id="{experiment_id}"',
                f'{load_source}.run_id="{run_id}"',
            ]
        )

    with initialize_config_dir(config_dir=CONFIG_DIR):
        expected = compose(config_name=f"{test_case}_expected", overrides=expected_overrides)

    assert actual == expected
