from pathlib import Path
import multiprocessing as mp
from epsilon_transformers.training.trainer import Trainer
from epsilon_transformers.training.configs import TrainConfig

def run_experiment(config: TrainConfig):
    """Run a single training experiment"""
    trainer = Trainer(config)
    trainer.train()

def run_parallel_experiments(configs: list[TrainConfig], num_processes: int | None = None):
    """Run multiple experiments in parallel"""
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=num_processes) as pool:
        pool.map(run_experiment, configs)

def create_run_folder(config: TrainConfig) -> Path:
    """Create uniquely named run folder based on config"""
    # TODO: Implement folder naming logic
    pass