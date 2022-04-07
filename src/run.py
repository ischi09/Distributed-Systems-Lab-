import random
import numpy as np
import hydra
import torch
import sys
from deepsets.config import Config
from deepsets.datasets import generate_datasets
from deepsets.networks import generate_model
from deepsets.experiments import Experiment


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config")
def main(config: Config):
    set_random_seeds(config.experiment.random_seed)

    train_set, valid_set, test_set = generate_datasets(config)

    print("Training Dataset Statistics:")
    print(f"\tlabel mean: {train_set.label_mean}")
    print(f"\tlabel std: {train_set.label_std}")
    print(f"\tlabel median: {train_set.label_median}")
    print(f"\tlabel mode: {train_set.label_mode}")
    print(f"\tlabel max: {train_set.label_max}")
    print(f"\tlabel min: {train_set.label_min}\n")

    model = generate_model(config.model, delta=train_set.delta)

    experiment = Experiment(
        config=config,
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
    )

    experiment.run()


if __name__ == "__main__":
    main()
