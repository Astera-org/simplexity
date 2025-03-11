import hydra
import jax

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train import train


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config):
    """Run the experiment."""
    print(cfg)

    key = jax.random.PRNGKey(cfg.seed)

    generative_process = typed_instantiate(cfg.generative_process.instance, GenerativeProcess)
    initial_gen_process_state = generative_process.initial_state
    num_observations = generative_process.num_observations

    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, in_size=1, out_size=num_observations)

    model, losses = train(
        cfg.train,
        key,
        model,
        generative_process,
        initial_gen_process_state,
        log_every=1,
    )
    print(losses.shape)


if __name__ == "__main__":
    run_experiment()
