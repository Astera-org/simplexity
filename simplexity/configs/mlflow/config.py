from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for MLflow."""

    experiment_name: str
    run_name: str
    tracking_uri: str
    registry_uri: str
    downgrade_unity_catalog: bool
