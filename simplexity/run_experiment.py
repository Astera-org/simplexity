import hydra
import jax

from simplexity.configs.config import Config
from simplexity.training.train import train


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config):
    """Run the experiment."""
    print(cfg)

    key = jax.random.PRNGKey(cfg.seed)

    generative_process = hydra.utils.instantiate(cfg.generative_process.instance)
    initial_gen_process_state = generative_process.state_eigenvector
    num_observations = generative_process.num_observations

    model = hydra.utils.instantiate(cfg.predictive_model.instance, in_size=1, out_size=num_observations)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer.instance)

    sequence_len = cfg.train.sequence_len
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs

    model, losses = train(
        key,
        model,
        optimizer,
        generative_process,
        initial_gen_process_state,
        num_epochs,
        batch_size,
        sequence_len,
        log_every=1,
    )
    print(losses.shape)


if __name__ == "__main__":
    run_experiment()
