from typing import Callable, List, Tuple, Iterator, Dict
import random

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import (
    Sampler,
    RandomSampler,
    SubsetRandomSampler,
)

from config import Config
from tasks import Task, ClassificationTask, get_task


class SetDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        max_set_size: int,
        min_value: int,
        max_value: int,
        use_multisets,
        sample_set: Callable[[int, int, int, bool], np.ndarray],
        task: Task,
    ):
        self.n_samples = n_samples
        self.sets = []
        # Map from set size to indices in sets list that have this size.
        self.set_distribution: Dict[int, List[int]] = {}

        for i in range(n_samples):
            # Generate the actual random set.
            rand_set = sample_set(
                min_value, max_value, max_set_size, use_multisets
            )
            rand_set_size = rand_set.shape[0]

            # Generate the padding to the maximum size.
            padding_len = max_set_size - rand_set_size
            padding = np.repeat(task.padding_element(), padding_len)
            padding = padding[:, np.newaxis]

            # Generate padded random set and padding mask.
            padded_rand_set = np.vstack((rand_set, padding))
            mask = np.vstack(
                (
                    np.ones(rand_set.shape, dtype=np.int),
                    np.zeros(padding.shape, dtype=np.int),
                )
            )

            def to_tensor(x: np.ndarray) -> torch.Tensor:
                return torch.tensor(x, dtype=torch.float)

            rand_set = to_tensor(rand_set)
            padded_rand_set = to_tensor(padded_rand_set)
            mask = to_tensor(mask)

            self.sets.append(
                (padded_rand_set, mask, task.generate_label(rand_set))
            )
            try:
                self.set_distribution[rand_set_size].append(i)
            except KeyError:
                self.set_distribution[rand_set_size] = []
                self.set_distribution[rand_set_size].append(i)

        # After processing for later evaluation
        if isinstance(task, ClassificationTask):
            labels = [torch.argmax(label) for _, _, label in self.sets]
        else:
            labels = [label for _, _, label in self.sets]
        self.label_mean = np.mean(labels)
        self.label_mode = float(torch.tensor(labels).squeeze().mode().values)
        self.label_median = float(np.median(labels))
        self.label_max = float(np.max(labels))
        self.label_min = float(np.min(labels))
        self.label_std = float(np.std(labels))

        # Delta for PNA architecture.
        set_degrees = [mask.sum() + 1 for _, mask, _, in self.sets]
        log_set_degrees = np.log(set_degrees)
        self.delta = float(np.mean(log_set_degrees))

        # Class weights for weighted cross-entropy.
        y = np.array(labels)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        self.class_weights = torch.from_numpy(class_weights)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sets[idx]


class BatchSetSampler(Sampler[List[int]]):
    def __init__(
        self, dataset: SetDataset, batch_size: int, generator=None
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        # Precompute random assignments of equal-length sets into batches. This
        # ensures that a single batch has sets of all the same lengths.
        batches = []
        for set_indices in self.dataset.set_distribution.values():
            set_index_sampler = SubsetRandomSampler(
                indices=set_indices, generator=self.generator
            )
            batch = []
            for idx in set_index_sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    batches.append(batch[:])
                    batch = []
            if len(batch) > 0:
                batches.append(batch[:])

        # Yield batches in random order. This ensures that two batches sampled
        # after each other will likely have sets of different lengths while
        # within the batch have sets of all the same length.
        batch_sampler = RandomSampler(
            data_source=batches, generator=self.generator
        )
        for idx in batch_sampler:
            yield batches[idx]

    def __len__(self) -> int:
        n_batches = 0
        for set_indices in self.dataset.set_distribution.values():
            n_full_batches = len(set_indices) // self.batch_size
            n_partial_batches = (
                0 if len(set_indices) % self.batch_size == 0 else 1
            )
            n_batches += n_full_batches + n_partial_batches
        return n_batches


def sample_integer_set(
    min_value: int, max_value: int, max_set_size: int, use_multisets: bool
) -> np.ndarray:
    values = np.arange(min_value, max_value + 1)
    rand_set_size = random.randint(1, max_set_size)
    rand_set_shape = (rand_set_size, 1)
    rand_set = np.random.choice(values, rand_set_shape, replace=use_multisets)
    return rand_set


def sample_longest_len_set(
    min_value: int, max_value: int, max_set_size: int, use_multisets: bool
) -> np.ndarray:
    values = np.arange(min_value, max_value + 1)
    set_len = random.randint(1, max_set_size)
    long_len = random.randint(1, set_len)
    start = random.randint(0, len(values) - 1 - long_len)
    sequence = values[start : start + long_len]

    set_shape = (set_len - long_len, 1)
    if use_multisets:
        rand_vals = np.random.choice(values, set_shape, replace=use_multisets)
    else:
        remaining_vals = np.array(
            [val for val in values if val not in sequence]
        )
        rand_vals = np.random.choice(
            remaining_vals, set_shape, replace=use_multisets
        )

    final = np.concatenate((sequence[..., np.newaxis], rand_vals))
    np.random.shuffle(final)
    return final


def build_datasets(config: Config):
    if config.task.label == "longest_seq_length":
        sample_set = sample_longest_len_set
    else:
        sample_set = sample_integer_set

    task = get_task(config.task)

    train_set = SetDataset(
        n_samples=config.datasets.train_samples,
        max_set_size=config.task.max_set_size,
        min_value=config.task.min_value,
        max_value=config.task.max_value,
        use_multisets=config.task.multisets,
        sample_set=sample_set,
        task=task,
    )

    valid_set = SetDataset(
        n_samples=config.datasets.valid_samples,
        max_set_size=config.task.max_set_size,
        min_value=config.task.min_value,
        max_value=config.task.max_value,
        use_multisets=config.task.multisets,
        sample_set=sample_set,
        task=task,
    )

    test_set = SetDataset(
        n_samples=config.datasets.test_samples,
        max_set_size=config.task.max_set_size,
        min_value=config.task.min_value,
        max_value=config.task.max_value,
        use_multisets=config.task.multisets,
        sample_set=sample_set,
        task=task,
    )

    return train_set, valid_set, test_set


def get_data_loader(
    dataset: SetDataset, batch_size: int, use_batch_sampler: bool
) -> DataLoader:
    if use_batch_sampler:
        batch_sampler = BatchSetSampler(dataset, batch_size)
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    else:
        data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )
    return data_loader
