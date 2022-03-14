from typing import Tuple

import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset


class DummySummation(Dataset):
    def __init__(self, is_train: bool = True):
        if is_train:
            self.items = torch.rand(10000, 10, 1)
        else:
            self.items = torch.rand(1000, 10, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        return self.items[item], self.items[item].sum()


class DummyMax(Dataset):
    def __init__(self, is_train: bool = True):
        if is_train:
            self.items = torch.rand(10000, 10, 1)
        else:
            self.items = torch.rand(1000, 10, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        return self.items[item], self.items[item].max()


class DummyCardinality(Dataset):
    def __init__(self, is_train: bool = True):
        if is_train:
            self.items = torch.rand(10000, 10, 1)
        else:
            self.items = torch.rand(1000, 10, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        return self.items[item], torch.tensor(
            len(self.items[item]),
            dtype=torch.float)
