from pathlib import Path

from simplexity.persistence.local_persister import LocalPersister

try:
    import torch
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


class LocalPytorchPersister(LocalPersister):
    """Persists a PyTorch model to the local filesystem."""

    filename: str = "model.pt"

    def __init__(self, directory: str | Path, filename: str = "model.pt"):
        self.directory = Path(directory)
        self.filename = filename

    # TODO: This is a hack to get the type checker to work.
    def save_weights(self, model: torch.nn.Module, step: int = 0, overwrite_existing: bool = False) -> None:  # type: ignore
        """Saves a PyTorch model to the local filesystem."""
        path = self._get_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite_existing and path.exists():
            path.unlink()

        torch.save(model.state_dict(), path)

    # TODO: This is a hack to get the type checker to work.
    def load_weights(self, model: torch.nn.Module, step: int = 0) -> torch.nn.Module:  # type: ignore
        """Loads weights into a PyTorch model from the local filesystem."""
        path = self._get_path(step)
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / self.filename

    # --- Checkpoint discovery ---
    def list_checkpoints(self) -> list[int]:
        steps: list[int] = []
        if not self.directory.exists():
            return steps
        for child in self.directory.iterdir():
            if child.is_dir():
                try:
                    step = int(child.name)
                except ValueError:
                    continue
                if (child / self.filename).exists():
                    steps.append(step)
        steps.sort()
        return steps

    def latest_checkpoint(self) -> int | None:
        steps = self.list_checkpoints()
        return steps[-1] if steps else None

    def checkpoint_exists(self, step: int) -> bool:
        return (self.directory / str(step) / self.filename).exists()

    def uri_for_step(self, step: int) -> str:
        path = self._get_path(step)
        return f"file://{path}"
