import tempfile
from pathlib import Path
from typing import Protocol

import boto3
import equinox as eqx
from botocore.exceptions import ClientError

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel


class S3Client(Protocol):
    """Protocol for S3 client.

    Since boto3 does not currently support type checking.
    https://github.com/boto/boto3/issues/1055
    """

    def upload_file(self, file_name: str, bucket: str, object_name: str) -> None:
        """Upload a file to S3."""
        ...

    def download_file(self, bucket: str, object_name: str, file_name: str) -> None:
        """Download a file from S3."""
        ...


class S3ModelPersister(ModelPersister):
    """Persists a model to an S3 bucket."""

    bucket: str
    prefix: str
    s3_client: S3Client

    @classmethod
    def from_client_args(
        cls,
        bucket: str,
        prefix: str = "models",
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> "S3ModelPersister":
        """Creates a new S3ModelPersister from client arguments."""
        prefix = prefix.rstrip("/")
        s3_client = boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url)
        return cls(bucket=bucket, prefix=prefix, s3_client=s3_client)  # type: ignore

    def save_weights(self, model: PredictiveModel, name: str) -> None:
        """Saves a model to S3."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_name = self._get_file_name(name)
                path = Path(temp_dir) / file_name
                eqx.tree_serialise_leaves(path, model)
                object_name = f"{self.prefix}/{file_name}"
                self.s3_client.upload_file(str(path), self.bucket, object_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise RuntimeError(f"Bucket {self.bucket} does not exist") from e
            elif error_code == "AccessDenied":
                raise RuntimeError(f"Access denied to bucket {self.bucket}") from e
            else:
                raise RuntimeError(f"Failed to save model to S3: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving model to S3: {e}") from e

    def load_weights(self, model: PredictiveModel, name: str) -> PredictiveModel:
        """Loads a model from S3."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_name = self._get_file_name(name)
                object_name = f"{self.prefix}/{file_name}"
                path = Path(temp_dir) / file_name
                self.s3_client.download_file(self.bucket, object_name, str(path))
                return eqx.tree_deserialise_leaves(path, model)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise RuntimeError(f"Model {name} not found in bucket {self.bucket}") from e
            elif error_code == "NoSuchBucket":
                raise RuntimeError(f"Bucket {self.bucket} does not exist") from e
            elif error_code == "AccessDenied":
                raise RuntimeError(f"Access denied to bucket {self.bucket}") from e
            else:
                raise RuntimeError(f"Failed to load model from S3: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading model from S3: {e}") from e

    def _get_file_name(self, name: str) -> str:
        """Constructs the full file name for a model."""
        if not name.endswith(".eqx"):
            name = f"{name}.eqx"
        return name
