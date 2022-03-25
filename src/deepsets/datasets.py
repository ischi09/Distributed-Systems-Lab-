from typing import Callable, Any
import random
from .config import Config
import hydra
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def get_max(x: torch.Tensor):
    return x.max()


def get_mode(x: torch.Tensor):
    return torch.squeeze(x).mode().values


def get_cardinality(x: torch.Tensor):
    return torch.tensor(len(x), dtype=torch.float)


def get_sum(x: torch.Tensor):
    return x.sum()



LABEL_GENERATORS = {
    "sum": get_sum,
    "cardinality": get_cardinality,
    "mode": get_mode,
    "max": get_max,
}


def generate_datasets(cfg: Config):
    train = cfg.trainset
    valid = cfg.validset
    test = cfg.testset

    train_set = SetDataset(
        train.n_samples,
        train.max_set_size,
        train.min_value,
        train.max_value,
        LABEL_GENERATORS[train.label],
        train.multisets,
    )
    valid_set = SetDataset(
        valid.n_samples,
        valid.max_set_size,
        valid.min_value,
        valid.max_value,
        LABEL_GENERATORS[valid.label],
        valid.multisets,
    )
    test_set = SetDataset(
        test.n_samples,
        test.max_set_size,
        test.min_value,
        test.max_value,
        LABEL_GENERATORS[test.label],
        test.multisets,
    )

    return train_set, valid_set, test_set


class SetDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        max_set_size: int,
        min_value: int,
        max_value: int,
        label_generator: Callable[[Any], Any],
        generate_multisets=False,
    ):
        self.n_samples = n_samples
        self.label_generator = label_generator
        self.items = []

        for _ in range(n_samples):
            values = np.arange(min_value, max_value)
            rand_set_size = random.randint(1, max_set_size)
            rand_set = np.random.choice(
                values, (rand_set_size, 1), replace=generate_multisets
            )
            item = torch.tensor(rand_set, dtype=torch.float)
            self.items.append(item)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        elem = self.items[item]
        return elem, self.label_generator(elem)
