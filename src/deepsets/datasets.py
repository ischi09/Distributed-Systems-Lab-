from typing import Callable, Any, Tuple
import random

from .config import Config
import hydra
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from functools import partial


def get_max(x: torch.Tensor) -> torch.Tensor:
    return x.max()


def get_mode(x: torch.Tensor) -> torch.Tensor:
    return torch.squeeze(x).mode().values


def get_cardinality(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(len(x), dtype=torch.float)


def get_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum()


def get_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean()


def get_largest_seq_length(x: torch.Tensor) -> torch.Tensor:
    sorted_, _ = torch.sort(x)
    max_length = 0
    cur_length = 0
    last_val = None
    for val in sorted_:
        if last_val is None:
            last_val = val
            continue

        if last_val + 1 == val:
            cur_length += 1
        else:
            max_length = max(max_length, cur_length)
            cur_length = 0

    return torch.Tensor(max_length)


def get_largest_contiguous_sum(x: torch.Tensor) -> torch.Tensor:
    sorted_, _ = torch.sort(x, descending=True)
    total = 0

    for val in sorted_:
        if val < 0:
            break

        total += val

    return torch.Tensor(total)


def get_largest_n_tuple_sum(x: torch.Tensor, n: int) -> torch.Tensor:
    sorted_, _ = torch.sort(x, descending=True)
    total = 0

    for i in range(n):
        total += sorted_[i]

    return torch.Tensor(total)


LABEL_GENERATORS = {
    "sum": get_sum,
    "cardinality": get_cardinality,
    "mode": get_mode,
    "max": get_max,
    "mean": get_mean,
    "largest_seq_length": get_largest_seq_length,
    "largest_contiguous_sum": get_largest_contiguous_sum,
    "largest_pair_sum": partial(get_largest_n_tuple_sum, n=2),
    "largest_triple_sum": partial(get_largest_n_tuple_sum, n=3),
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
