from simplexity.configs.train.config import Config as TrainConfig
from simplexity.configs.train.optimizer.config import AdamConfig
from simplexity.configs.train.optimizer.config import Config as OptimizerConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.training.train import train


def test_train():
    generative_process = build_hidden_markov_model("even_ones", p=0.5)
    initial_gen_process_state = generative_process.stationary_state
    model = build_gru_rnn(generative_process.vocab_size, num_layers=2, hidden_size=4, seed=0)

    cfg = TrainConfig(
        seed=0,
        sequence_len=4,
        batch_size=2,
        num_steps=8,
        log_every=1,
        optimizer=OptimizerConfig(
            name="adam",
            instance=AdamConfig(
                _target_="optax.adam",
                learning_rate=0.001,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                eps_root=0.0,
                nesterov=True,
            ),
        ),
    )
    model, losses = train(cfg, model, generative_process, initial_gen_process_state)
    assert losses.shape == (cfg.num_steps,)
