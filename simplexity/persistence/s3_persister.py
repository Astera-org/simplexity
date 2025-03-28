import tempfile
from configparser import ConfigParser
from pathlib import Path
from typing import Protocol

import equinox as eqx
from boto3.session import Session
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


class S3Persister(ModelPersister):
    """Persists a model to an S3 bucket."""

    bucket: str
    prefix: str
    s3_client: S3Client

    @classmethod
    def from_config(cls, filename: str) -> "S3Persister":
        """Creates a new S3Persister from client arguments."""
        config = ConfigParser()
        config.read(filename)
        bucket = config["s3"]["bucket"]
        prefix = config["s3"]["prefix"]
        profile_name = config["aws"]["profile_name"]
        session = Session(profile_name=profile_name)
        s3_client = session.client("s3")
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
