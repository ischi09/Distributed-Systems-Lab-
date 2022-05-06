import os
import random
import numpy as np
import hydra
import torch

from config import Config
from datasets import generate_datasets
from models import generate_model
from experiments import Experiment


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(
    config_path=os.path.join(os.pardir, "config"), config_name="config"
)
def main(config: Config) -> None:
    set_random_seeds(config.experiment.random_seed)

    train_set, valid_set, test_set = generate_datasets(config)

    model = generate_model(
        config=config,
        delta=train_set.delta,
    )

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
