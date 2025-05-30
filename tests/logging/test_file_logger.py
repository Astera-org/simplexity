from pathlib import Path

import jax.numpy as jnp
from omegaconf import DictConfig

from simplexity.logging.file_logger import FileLogger

EXPECTED_LOG = """Config: {'str_param': 'str_value', 'int_param': 1, 'float_param': 1.0, 'bool_param': True}
Params: {'str_param': 'str_value', 'int_param': 1, 'float_param': 1.0, 'bool_param': True}
Tags: {'str_tag': 'str_value', 'int_tag': 1, 'float_tag': 1.0, 'bool_tag': True}
Metrics at step 1: {'int_metric': 1, 'float_metric': 1.0, 'jnp_metric': Array(0.1, dtype=float32, weak_type=True)}
"""


def test_file_logger(tmp_path: Path):
    logger = FileLogger(str(tmp_path / "test.log"))
    params = {
        "str_param": "str_value",
        "int_param": 1,
        "float_param": 1.0,
        "bool_param": True,
    }
    logger.log_config(DictConfig(params))
    logger.log_params(params)
    tags = {
        "str_tag": "str_value",
        "int_tag": 1,
        "float_tag": 1.0,
        "bool_tag": True,
    }
    logger.log_tags(tags)
    metrics = {
        "int_metric": 1,
        "float_metric": 1.0,
        "jnp_metric": jnp.array(0.1),
    }
    logger.log_metrics(1, metrics)
    logger.close()

    with open(tmp_path / "test.log") as f:
        assert f.read() == EXPECTED_LOG
