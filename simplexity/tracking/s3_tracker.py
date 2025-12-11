"""S3 tracker."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import configparser
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig

from simplexity.predictive_models.types import ModelFramework
from simplexity.tracking.model_persistence.local_model_persister import (
    LocalModelPersister,
)
from simplexity.tracking.tracker import RunTracker
from simplexity.tracking.utils import build_local_persister


class S3Paginator(Protocol):
    """Protocol for an S3 paginator."""

    def paginate(self, Bucket: str, Prefix: str) -> Iterable[Mapping[str, Any]]:  # pylint: disable=invalid-name
        """Paginate over the objects in an S3 bucket."""
        ...


class S3Client(Protocol):
    """Protocol for S3 client."""

    def upload_file(self, file_name: str, bucket: str, object_name: str) -> None:
        """Upload a file to S3."""

    def download_file(self, bucket: str, object_name: str, file_name: str) -> None:
        """Download a file from S3."""

    def get_paginator(self, operation_name: str) -> S3Paginator:
        """Get a paginator for the given operation."""
        ...


class S3Tracker(RunTracker):
    """Tracks runs to S3 (persistence only)."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        s3_client: S3Client,
        temp_dir: tempfile.TemporaryDirectory,
        local_persisters: dict[ModelFramework, LocalModelPersister] | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = s3_client
        self.temp_dir = temp_dir
        self.local_persisters = local_persisters or {}

    @classmethod
    def from_config(
        cls,
        prefix: str,
        config_filename: str = "config.ini",
    ) -> "S3Tracker":
        """Creates a new S3Tracker from configuration parameters."""
        import boto3.session  # pylint: disable=import-outside-toplevel

        config = configparser.ConfigParser()
        config.read(config_filename)

        bucket = config.get("s3", "bucket")
        profile_name = config.get("aws", "profile_name", fallback="default")
        session = boto3.session.Session(profile_name=profile_name)
        s3_client = session.client("s3")
        temp_dir = tempfile.TemporaryDirectory()

        return cls(
            bucket=bucket,
            prefix=prefix,
            s3_client=s3_client,  # type: ignore
            temp_dir=temp_dir,
        )

    # Lifecycle

    def cleanup(self) -> None:
        """Cleans up the temporary directory."""
        self.temp_dir.cleanup()

    # Logging (Not Implemented / No-ops)

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config (Not Supported)."""
        pass

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics (Not Supported)."""
        pass

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params (Not Supported)."""
        pass

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags (Not Supported)."""
        pass

    def log_figure(self, figure: Any, artifact_file: str, **kwargs) -> None:
        """Log figure (Not Supported)."""
        pass

    def log_image(
        self,
        image: Any,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log image (Not Supported)."""
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log artifact (Not Supported)."""
        pass

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Log JSON artifact (Not Supported)."""
        pass

    # Persistence

    def save_model(self, model: Any, step: int = 0) -> None:
        """Saves a model to S3."""
        from simplexity.predictive_models.types import get_model_framework

        framework = get_model_framework(model)

        if framework not in self.local_persisters:
            # Build one in the temp dir
            self.local_persisters[framework] = build_local_persister(framework, Path(self.temp_dir.name))

        local_persister = self.local_persisters[framework]
        local_persister.save_weights(model, step)
        directory = local_persister.directory / str(step)
        self._upload_local_directory(directory)

    def load_model(self, model: Any, step: int = 0) -> Any:
        """Loads a model from S3."""
        from simplexity.predictive_models.types import get_model_framework

        framework = get_model_framework(model)

        if framework not in self.local_persisters:
            self.local_persisters[framework] = build_local_persister(framework, Path(self.temp_dir.name))
        local_persister = self.local_persisters[framework]

        self._download_s3_objects(step, local_persister)
        return local_persister.load_weights(model, step)

    def _upload_local_directory(self, directory: Path) -> None:
        for root, _, files in directory.walk():
            for file in files:
                file_path = root / file
                relative_path = file_path.relative_to(directory.parent)
                object_name = f"{self.prefix}/{relative_path}"
                file_name = str(file_path)
                self._upload_local_file(file_name, object_name)

    def _upload_local_file(self, file_name: str, object_name: str) -> None:
        from botocore.exceptions import ClientError

        try:
            self.s3_client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise RuntimeError(f"Bucket {self.bucket} does not exist") from e
            elif error_code == "AccessDenied":
                raise RuntimeError(f"Access denied to bucket {self.bucket}") from e
            else:
                raise RuntimeError(f"Failed to save {file_name} to S3: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving {file_name} to S3: {e}") from e

    def _download_s3_objects(self, step: int, local_persister: LocalModelPersister) -> None:
        prefix = f"{self.prefix}/{step}"
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                object_name = obj["Key"]
                relative_path = Path(object_name).relative_to(self.prefix)
                file_name = str(local_persister.directory / relative_path)
                self._download_s3_object(object_name, file_name)

    def _download_s3_object(self, object_name: str, file_name: str) -> None:
        from botocore.exceptions import ClientError

        try:
            local_path = Path(file_name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, object_name, file_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise RuntimeError(f"{file_name} not found in bucket {self.bucket}") from e
            elif error_code == "NoSuchBucket":
                raise RuntimeError(f"Bucket {self.bucket} does not exist") from e
            elif error_code == "AccessDenied":
                raise RuntimeError(f"Access denied to bucket {self.bucket}") from e
            else:
                raise RuntimeError(f"Failed to load {file_name} from S3: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {file_name} from S3: {e}") from e

    # Model Registry (Not supported)
    def save_model_to_registry(self, model: Any, registered_model_name: str, **kwargs) -> Any:
        """Save a model to the registry (Not Supported)."""
        raise NotImplementedError("S3Tracker does not support model registry.")

    def load_model_from_registry(self, registered_model_name: str, **kwargs) -> Any:
        """Load a model from the registry (Not Supported)."""
        raise NotImplementedError("S3Tracker does not support model registry.")
