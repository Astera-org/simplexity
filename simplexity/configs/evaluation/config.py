from simplexity.run_management.structured_configs import (
    ValidationConfig as Config,
    validate_validation_config as validate_config,
)

# Re-export for backwards compatibility
__all__ = ["Config", "validate_config"]
