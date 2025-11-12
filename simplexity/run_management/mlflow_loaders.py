import logging
from typing import Any

from omegaconf import OmegaConf

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.run_management.components import Components
from simplexity.run_management.run_management import _get_config, _setup

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


def load_run_components(
    run_id: str,
    tracking_uri: str | None = None,
    config_keys: dict[str, str] | None = None,
) -> Components:
    """Load components from an Mlflow run using the existing config loading infrastructure."""
    if config_keys is None:
        config_keys = {
            "predictive_model": "predictive_model",
            "generative_process": "generative_process",
        }

    load_config = {
        "run_id": run_id,
        "configs": config_keys,
    }

    if tracking_uri is not None:
        load_config["tracking_uri"] = tracking_uri

    cfg_dict = {
        "load_configs": [load_config],
    }

    cfg = OmegaConf.create(cfg_dict)

    SIMPLEXITY_LOGGER.info("[mlflow loader] loading components from run '%s'", run_id)

    loaded_cfg = _get_config((cfg,), {})

    components = _setup(loaded_cfg, strict=False, verbose=False)

    SIMPLEXITY_LOGGER.info("[mlflow loader] successfully loaded components")
    return components


def load_model_and_generative_process(
    run_id: str,
    tracking_uri: str | None = None,
    model_key: str = "predictive_model",
    generative_process_key: str = "generative_process",
    persister_key: str | None = None,
) -> tuple[Any, GenerativeProcess]:
    """Load model and generative process from an Mlflow run."""
    config_keys = {
        model_key: "predictive_model",
        generative_process_key: "generative_process",
    }

    if persister_key is not None:
        config_keys[persister_key] = "persistence"

    components = load_run_components(run_id=run_id, tracking_uri=tracking_uri, config_keys=config_keys)

    model = components.get_predictive_model()
    if model is None:
        raise ValueError("No predictive model found in loaded components")

    generative_process = components.get_generative_process()
    if generative_process is None:
        raise ValueError("No generative process found in loaded components")

    return model, generative_process
