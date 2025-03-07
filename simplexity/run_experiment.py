import hydra
import jax
import optax

from simplexity.configs.config import Config
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import mess3
from simplexity.training.train import train


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config):
    """Run the experiment."""
    print(cfg)

    key = jax.random.PRNGKey(cfg.seed)

    transition_matrices = mess3()
    vocab_size = transition_matrices.shape[0]
    generative_process = HiddenMarkovModel(transition_matrices)
    initial_gen_process_state = generative_process.state_eigenvector

    model = hydra.utils.instantiate(cfg.predictive_model.instance, in_size=1, out_size=vocab_size)
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
