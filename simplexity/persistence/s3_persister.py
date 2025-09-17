import configparser
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import boto3.session
from botocore.exceptions import ClientError

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.persistence.local_persister import LocalPersister
from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.predictive_models.types import ModelFramework


class S3Paginator(Protocol):
    """Protocol for an S3 paginator.

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#paginators
    Since boto3 does not currently support type checking: https://github.com/boto/boto3/issues/1055
    """

    def paginate(self, Bucket: str, Prefix: str) -> Iterable[Mapping[str, Any]]:
        """Paginate over the objects in an S3 bucket."""
        ...


class S3Client(Protocol):
    """Protocol for S3 client.

    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client
    Since boto3 does not currently support type checking: https://github.com/boto/boto3/issues/1055
    """

    def upload_file(self, file_name: str, bucket: str, object_name: str) -> None:
        """Upload a file to S3."""
        ...

    def download_file(self, bucket: str, object_name: str, file_name: str) -> None:
        """Download a file from S3."""
        ...

    def get_paginator(self, operation_name: str) -> S3Paginator:
        """Get a paginator for the given operation."""
        ...


class S3Persister(ModelPersister):
    """Persists a model to an S3 bucket."""

    bucket: str
    prefix: str
    s3_client: S3Client
    temp_dir: tempfile.TemporaryDirectory
    local_persister: LocalPersister

    @classmethod
    def from_config(
        cls,
        prefix: str,
        model_framework: ModelFramework = ModelFramework.Equinox,
        config_filename: str = "config.ini",
    ) -> "S3Persister":
        """Creates a new S3Persister from configuration parameters.

        Args:
            prefix: S3 prefix for model storage (from YAML config)
            model_framework: Framework for local persistence
            config_filename: Path to config.ini file containing AWS settings
        """
        config = configparser.ConfigParser()
        config.read(config_filename)

        bucket = config.get("s3", "bucket")
        profile_name = config.get("aws", "profile_name", fallback="default")
        session = boto3.session.Session(profile_name=profile_name)
        s3_client = session.client("s3")
        temp_dir = tempfile.TemporaryDirectory()
        if model_framework == ModelFramework.Equinox:
            local_persister = LocalEquinoxPersister(directory=temp_dir.name)
        elif model_framework == ModelFramework.Penzai:
            local_persister = LocalPenzaiPersister(directory=temp_dir.name)
        elif model_framework == ModelFramework.Pytorch:
            local_persister = LocalPytorchPersister(directory=temp_dir.name)
        else:
            raise ValueError(f"Unsupported model framework: {model_framework}")

        return cls(
            bucket=bucket,
            prefix=prefix,
            s3_client=s3_client,  # type: ignore
            temp_dir=temp_dir,
            local_persister=local_persister,
        )

    def cleanup(self) -> None:
        """Cleans up the temporary directory."""
        self.temp_dir.cleanup()

    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Saves a model to S3."""
        self.local_persister.save_weights(model, step)
        directory = self.local_persister.directory / str(step)
        self._upload_local_directory(directory)

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from S3."""
        self._download_s3_objects(step)
        return self.local_persister.load_weights(model, step)

    def _upload_local_directory(self, directory: Path) -> None:
        for root, _, files in directory.walk():
            for file in files:
                file_path = root / file
                relative_path = file_path.relative_to(directory.parent)
                object_name = f"{self.prefix}/{relative_path}"
                file_name = str(file_path)
                self._upload_local_file(file_name, object_name)

    def _upload_local_file(self, file_name: str, object_name: str) -> None:
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

    def _download_s3_objects(self, step: int) -> None:
        prefix = f"{self.prefix}/{step}/"
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                object_name = obj["Key"]
                relative_path = Path(object_name).relative_to(self.prefix)
                file_name = str(self.local_persister.directory / relative_path)
                self._download_s3_object(object_name, file_name)

    def _download_s3_object(self, object_name: str, file_name: str) -> None:
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
