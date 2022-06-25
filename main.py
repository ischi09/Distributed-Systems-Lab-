import os
import random
import numpy as np
import hydra
import torch

from setml.config import Config
from setml.datasets import build_datasets
from setml.trainers import build_trainer
from setml.experiments import Experiment


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path=os.path.join("config"), config_name="config")
def main(config: Config) -> None:
    set_random_seeds(config.experiment.random_seed)

    train_set, valid_set, test_set = build_datasets(config)

    trainer = build_trainer(config=config, train_set=train_set)

    experiment = Experiment(
        config=config,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        trainer=trainer,
    )

    experiment.run()


if __name__ == "__main__":
    main()
