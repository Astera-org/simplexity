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

    def save_weights(self, model: torch.nn.Module, step: int = 0, overwrite_existing: bool = False) -> None:
        """Saves a PyTorch model to the local filesystem."""
        path = self._get_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if overwrite_existing and path.exists():
            path.unlink()
            
        torch.save(model.state_dict(), path)

    def load_weights(self, model: torch.nn.Module, step: int = 0) -> torch.nn.Module:
        """Loads weights into a PyTorch model from the local filesystem."""
        path = self._get_path(step)
        state_dict = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        model.load_state_dict(state_dict)
        return model

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / self.filename