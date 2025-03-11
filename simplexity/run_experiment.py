import hydra
import jax
from optax import GradientTransformation

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.persistence.model_persister import ModelPersister
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

    persister = typed_instantiate(cfg.persistence.instance, ModelPersister)
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, in_size=1, out_size=num_observations)
    if cfg.predictive_model.load_weights:
        model = persister.load_weights(model, cfg.predictive_model.weights_filename)
    optimizer = typed_instantiate(cfg.train.optimizer.instance, GradientTransformation)

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
    if cfg.train.save_weights:
        persister.save_weights(model, cfg.predictive_model.weights_filename)
    print(losses.shape)


if __name__ == "__main__":
    run_experiment()
