from typing import Callable, Any, Tuple
import random

from .config import Config
import hydra
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def get_max(x: torch.Tensor) -> torch.Tensor:
    return x.max()


def get_mode(x: torch.Tensor) -> torch.Tensor:
    return torch.squeeze(x).mode().values


def get_cardinality(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(len(x), dtype=torch.float)


def get_sum(x: torch.Tensor) -> torch.Tensor:
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
        self.sets = []

        for _ in range(n_samples):
            # Generate the actual random set.
            values = np.arange(min_value, max_value)
            rand_set_size = random.randint(1, max_set_size)
            rand_set_shape = (rand_set_size, 1)
            rand_set = np.random.choice(
                values, rand_set_shape, replace=generate_multisets
            )

            # Generate the padding to the maximum size.
            padding_shape = (max_set_size - rand_set_size, 1)
            padding = np.zeros(padding_shape, dtype=np.int)

            # Generate padded random set and padding mask.
            padded_rand_set = np.vstack((rand_set, padding))
            mask = np.vstack(
                (
                    np.ones(rand_set_shape, dtype=np.int),
                    np.zeros(padding_shape, dtype=np.int),
                )
            )

            def to_tensor(x: np.ndarray) -> torch.FloatTensor:
                return torch.tensor(x, dtype=torch.float)

            rand_set = to_tensor(rand_set)
            padded_rand_set = to_tensor(padded_rand_set)
            mask = to_tensor(mask)

            self.sets.append((padded_rand_set, label_generator(rand_set), mask))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self.sets[idx]
