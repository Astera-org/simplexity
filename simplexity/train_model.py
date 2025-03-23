import hydra

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train import train


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base="1.2")
def run_experiment(cfg: Config):
    """Run the experiment."""
    generative_process = typed_instantiate(cfg.generative_process.instance, GenerativeProcess)
    initial_gen_process_state = generative_process.initial_state
    vocab_size = generative_process.vocab_size
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, vocab_size=vocab_size)
    train(cfg.train, model, generative_process, initial_gen_process_state)
    print("Training complete")


if __name__ == "__main__":
    run_experiment()
