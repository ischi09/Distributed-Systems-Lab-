import numpy as np
import hydra
import torch
import sys
from deepsets.config import Config
from deepsets.datasets import generate_datasets
from deepsets.networks import DeepSetsInvariant, MLP
from deepsets.experiments import Experiment


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config")
def main(config: Config):
    set_random_seeds(config.experiment.random_seed)

    train_set, valid_set, test_set = generate_datasets(config)

    # Set up model.
    model = DeepSetsInvariant(
        phi=MLP(input_dim=1, hidden_dim=10, output_dim=config.model.laten_dim),
        rho=MLP(input_dim=config.model.laten_dim, hidden_dim=10, output_dim=1),
        accumulator=config.model.accumulator,
    )

    experiment = Experiment(
        config=config,
        model=model,
        model_name="deepsets_invariant_mlp_sum_mlp",
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
    )

    experiment.run()


if __name__ == "__main__":
    main()
