from pathlib import Path

from omegaconf import DictConfig

from simplexity.logging.file_logger import FileLogger
from simplexity.logging.run_context import RunContext


def test_run_context_with_file_logger_logs_provenance(tmp_path: Path) -> None:
    """RunContext with a FileLogger logs config, params, and tags and calls cleanup.

    This test avoids MLflow and Hydra dependencies by injecting a FileLogger and a
    dummy persister. It checks that provenance methods are invoked and that
    cleanup is called on exit, matching test style used elsewhere.
    """
    # Prepare minimal cfg used for log_config/log_params
    cfg = DictConfig(
        {
            "seed": 42,
            "training": {"num_steps": 1},
            "model": {"name": "dummy"},
        }
    )

    # File-backed logger to inspect output deterministically
    log_path = tmp_path / "run.log"
    logger = FileLogger(str(log_path))

    # Dummy persister to validate cleanup is called
    class DummyPersister:
        def __init__(self) -> None:
            self.cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    persister = DummyPersister()

    # Use RunContext with tags; avoid source_relpath/hydra artifacts in this unit test
    with RunContext(
        cfg,
        logger=logger,
        persister=persister,  # type: ignore[arg-type]
        tags={"run.kind": "train", "task": "test_run_context"},
        log_hydra_artifacts=False,
        log_git_info=False,        # Skip git in unit test to keep output stable
        log_environment=False,     # Skip env logging for speed/stability
    ) as ctx:
        # Exercise helper; no assertions here, file contents checked below
        ctx.apply_standard_tags("train", "test_run_context", extras={"seed": 42})

    # Verify cleanup was called
    assert persister.cleaned is True

    # Verify log file contains expected provenance entries
    text = log_path.read_text()
    assert "Config:" in text
    assert "Params:" in text
    assert "Tags:" in text

