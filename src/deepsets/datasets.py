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


def get_longest_seq_length(x: torch.Tensor) -> torch.Tensor:
    """
    Returns length of longest sequence of consecutive numbers.

    Example:
        [2, 4, 1, 5, 7] -> 2 (because 1,2 or 4,5 are consecutive)
    """
    sorted_, _ = torch.sort(x)
    max_length = 1
    cur_length = 1
    last_val = None
    for val in sorted_:
        if last_val is None:
            last_val = val
            continue

        if last_val + 1 == val or last_val == val:
            cur_length += 1
        else:
            max_length = max(max_length, cur_length)
            cur_length = 1

        last_val = val

    return torch.tensor(max_length, dtype=torch.float)


def get_largest_contiguous_sum(x: torch.Tensor) -> torch.Tensor:
    sorted_, _ = torch.sort(x, descending=True)
    total = 0.0

    for val in sorted_:
        if val < 0:
            break

        total += float(val)

    return torch.tensor(total, dtype=torch.float)


def get_largest_n_tuple_sum(x: torch.Tensor, n: int) -> torch.Tensor:
    sorted_, _ = torch.sort(x, descending=True)
    return sorted_[:n].sum()


LABEL_GENERATORS = {
    "sum": get_sum,
    "cardinality": get_cardinality,
    "mode": get_mode,
    "max": get_max,
    "mean": get_mean,
    "longest_seq_length": get_longest_seq_length,
    "largest_contiguous_sum": get_largest_contiguous_sum,
    "largest_pair_sum": partial(get_largest_n_tuple_sum, n=2),
    "largest_triple_sum": partial(get_largest_n_tuple_sum, n=3),
}


def sample_integer_set(
    min_value: int, max_value: int, max_set_size: int, use_multisets: bool
) -> np.ndarray:
    values = np.arange(min_value, max_value)
    rand_set_size = random.randint(1, max_set_size)
    rand_set_shape = (rand_set_size, 1)
    rand_set = np.random.choice(values, rand_set_shape, replace=use_multisets)
    return rand_set


def generate_datasets(config: Config):
    train = config.trainset
    valid = config.validset
    test = config.testset

    train_set = SetDataset(
        n_samples=train.n_samples,
        max_set_size=train.max_set_size,
        min_value=train.min_value,
        max_value=train.max_value,
        use_multisets=train.multisets,
        sample_set=sample_integer_set,
        generate_label=LABEL_GENERATORS[train.label],
    )

    valid_set = SetDataset(
        n_samples=valid.n_samples,
        max_set_size=valid.max_set_size,
        min_value=valid.min_value,
        max_value=valid.max_value,
        use_multisets=valid.multisets,
        sample_set=sample_integer_set,
        generate_label=LABEL_GENERATORS[valid.label],
    )

    test_set = SetDataset(
        n_samples=test.n_samples,
        max_set_size=test.max_set_size,
        min_value=test.min_value,
        max_value=test.max_value,
        use_multisets=test.multisets,
        sample_set=sample_integer_set,
        generate_label=LABEL_GENERATORS[test.label],
    )

    return train_set, valid_set, test_set


class SetDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        max_set_size: int,
        min_value: int,
        max_value: int,
        use_multisets,
        sample_set: Callable[[int, int, int, bool], torch.Tensor],
        generate_label: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.n_samples = n_samples
        self.sets = []

        for _ in range(n_samples):
            # Generate the actual random set.
            rand_set = sample_set(
                min_value, max_value, max_set_size, use_multisets
            )
            rand_set_size = rand_set.shape[0]

            # Generate the padding to the maximum size.
            padding_shape = (max_set_size - rand_set_size, 1)
            padding = np.zeros(padding_shape, dtype=np.int)

            # Generate padded random set and padding mask.
            padded_rand_set = np.vstack((rand_set, padding))
            mask = np.vstack(
                (
                    np.ones(rand_set.shape, dtype=np.int),
                    np.zeros(padding_shape, dtype=np.int),
                )
            )

            def to_tensor(x: np.ndarray) -> torch.FloatTensor:
                return torch.tensor(x, dtype=torch.float)

            rand_set = to_tensor(rand_set)
            padded_rand_set = to_tensor(padded_rand_set)
            mask = to_tensor(mask)

            self.sets.append((padded_rand_set, generate_label(rand_set), mask))

        # After processing for later evaluation
        label_list = [labels for _, labels, _ in self.sets]
        self.label_mean = np.mean(label_list)
        self.label_mode = float(get_mode(torch.tensor(label_list)))
        self.label_median = float(np.median(label_list))
        self.label_max = float(np.max(label_list))
        self.label_min = float(np.min(label_list))
        self.label_std = float(np.std(label_list))

        # Delta for PNA architecture.
        set_degrees = [mask.sum() + 1 for _, _, mask in self.sets]
        log_set_degrees = np.log(set_degrees)
        self.delta = float(np.mean(log_set_degrees))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sets[idx]
