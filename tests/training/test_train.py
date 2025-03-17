import jax
import optax

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import even_ones
from simplexity.predictive_models.rnn import RNN
from simplexity.training.train import train


def test_train():
    key = jax.random.PRNGKey(0)

    transition_matrices = even_ones(p=0.5)
    vocab_size = int(transition_matrices.shape[0])
    gen_process = HiddenMarkovModel(transition_matrices)
    initial_gen_process_state = gen_process.stationary_state

    hidden_size = 4
    hidden_sizes = [hidden_size] * 2
    key, model_key = jax.random.split(key)
    model = RNN(in_size=vocab_size, out_size=vocab_size, hidden_sizes=hidden_sizes, key=model_key)
    optimizer = optax.adam(learning_rate=0.001)

    sequence_len = 4
    batch_size = 2
    num_epochs = 8
    _, losses = train(
        key, model, optimizer, gen_process, initial_gen_process_state, num_epochs, batch_size, sequence_len
    )
    assert losses.shape == (num_epochs,)
