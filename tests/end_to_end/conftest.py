"""End-to-end tests configuration."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import shutil
from pathlib import Path

import mlflow
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.utils.mlflow_utils import get_experiment, get_run

CONFIG_DIR = str(Path(__file__).parent / "mlflow_defaults_configs")
EXPERIMENT_NAME = "test_mlflow_defaults"


@pytest.fixture(scope="session")
def setup_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Setup function."""
    tmp_path = tmp_path_factory.mktemp("mlflow_defaults")
    config_path = tmp_path / "configs"
    shutil.copytree(f"{CONFIG_DIR}/setup", str(config_path))
    tracking_uri = f"sqlite:///{tmp_path.resolve()}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    def log_to_mlflow(cfg: DictConfig | None, config_names: list[str], run_name: str) -> None:
        """Save config."""
        experiment = get_experiment(experiment_name=EXPERIMENT_NAME)
        assert experiment is not None
        experiment_id = experiment.experiment_id
        run = get_run(run_name=run_name, experiment_id=experiment_id)
        assert run is not None
        run_id = run.info.run_id
        with mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            log_system_metrics=False,
        ):
            logger = MLFlowLogger(tracking_uri=tracking_uri)
            if cfg is not None:
                logger.log_config(cfg, resolve=False)
            for config_name in config_names:
                logger.log_artifact(local_path=str(config_path / config_name), artifact_path="subdir")

    def previous_run(config_name: str | None, config_names: list[str], run_name: str) -> None:
        """Previous run."""
        if config_name is None:
            cfg = None
        else:
            with initialize_config_dir(config_dir=str(config_path)):
                cfg = compose(config_name=config_name)
        log_to_mlflow(cfg, config_names, run_name)

    previous_run(config_name="prev_config_1", config_names=["special_1.yaml"], run_name="prev_run_1")
    previous_run(config_name="prev_config_2", config_names=["special_2.yaml"], run_name="prev_run_2")

    return tmp_path
