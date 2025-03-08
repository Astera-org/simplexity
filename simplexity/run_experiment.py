import hydra
import jax
import optax

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
    optimizer = optax.adam(learning_rate=0.001)

    sequence_len = 8
    batch_size = 4
    num_epochs = 3

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
