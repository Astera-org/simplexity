import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class MockS3Paginator:
    """Local filesystem implementation of S3 paginator for testing."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def paginate(self, Bucket: str, Prefix: str) -> Iterable[Mapping[str, Any]]:
        """Paginate over the objects in an S3 bucket."""
        bucket_dir = self.root_dir / Bucket
        bucket_dir.mkdir(exist_ok=True)
        files = []
        for obj in bucket_dir.rglob("*"):
            if obj.is_file():
                relative_path = obj.relative_to(bucket_dir)
                key = str(relative_path)
                if key.startswith(Prefix):
                    files.append({"Key": key})
        return [{"Contents": files}]


class MockS3Client:
    """Local filesystem implementation of S3 client for testing."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def upload_file(self, file_name: str, bucket: str, object_name: str) -> None:
        """Copy file to mock S3 storage."""
        bucket_dir = self.root_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        target_path = bucket_dir / object_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, target_path)

    def download_file(self, bucket: str, object_name: str, file_name: str) -> None:
        """Copy file from mock S3 storage."""
        source_path = self.root_dir / bucket / object_name
        if not source_path.exists():
            raise RuntimeError(f"File not found: {object_name}")
        shutil.copy(source_path, file_name)

    def get_paginator(self, operation_name: str) -> MockS3Paginator:
        """Get a paginator for the given operation."""
        return MockS3Paginator(self.root_dir)


class MockBoto3Session:
    """Mock Boto3 Session for testing."""

    def __init__(self, profile_name: str | None = None):
        self._root_dir: Path | None = None
        self._s3_client: MockS3Client | None = None

    @classmethod
    def create(cls, root_dir: Path | str | None = None) -> "MockBoto3Session":
        """Create a new mock Boto3 session."""
        session = cls()
        session.root_dir = root_dir
        return session

    @property
    def root_dir(self) -> Path:
        """Get the root directory for the mock S3 storage."""
        if not self._root_dir:
            self._root_dir = Path(tempfile.mkdtemp())
        return self._root_dir

    @root_dir.setter
    def root_dir(self, root_dir: Path | str | None) -> None:
        """Set the root directory for the mock S3 storage."""
        if self._s3_client:
            raise RuntimeError("Cannot set root directory after S3 client has been created")
        self._root_dir = Path(root_dir) if root_dir else None

    @property
    def s3_client(self) -> MockS3Client:
        """Get the S3 client for the mock S3 storage."""
        if not self._s3_client:
            self._s3_client = MockS3Client(self.root_dir)
        return self._s3_client

    def client(self, service_name: str) -> Any:
        """Get a client for the given service."""
        if service_name == "s3":
            return self.s3_client
        raise ValueError(f"Unsupported service: {service_name}")

    def cleanup(self) -> None:
        """Clean up the mock S3 storage."""
        if self._root_dir:
            shutil.rmtree(self._root_dir)
            self._root_dir = None
            self._s3_client = None
