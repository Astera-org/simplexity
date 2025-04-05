from simplexity.configs.validation.config import Config as ValidateConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.validation.validate_equinox_model import validate


def test_validate():
    cfg = ValidateConfig(seed=0, sequence_len=4, batch_size=2, num_steps=3, log_every=5)
    data_generator = build_hidden_markov_model("even_ones", p=0.5)
    model = build_gru_rnn(data_generator.vocab_size, num_layers=2, hidden_size=4, seed=0)
    metrics = validate(model, cfg, data_generator)
    assert metrics["loss"] > 0.0
    assert metrics["accuracy"] >= 0.0
    assert metrics["accuracy"] <= 1.0
