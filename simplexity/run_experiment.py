import jax
import optax

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import mess3
from simplexity.predictive_models.rnn import RNN
from simplexity.training.train import train


def run_experiment():
    """Run the experiment."""
    key = jax.random.PRNGKey(0)

    transition_matrices = mess3()
    vocab_size = transition_matrices.shape[0]
    generative_process = HiddenMarkovModel(transition_matrices)
    initial_gen_process_state = generative_process.state_eigenvector

    hidden_size = 64
    hidden_sizes = [hidden_size] * 4
    key, model_key = jax.random.split(key)
    model = RNN(in_size=1, out_size=vocab_size, hidden_sizes=hidden_sizes, key=model_key)
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
