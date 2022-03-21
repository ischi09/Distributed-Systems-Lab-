# Based on https://github.com/yassersouri/pytorch-deep-sets

from typing import Callable

import torch
import torch.nn as nn


def accumulate_sum(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.sum(axis=0)


def accumulate_max(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.max(dim=0).values


ACCUMLATORS = {"sum": accumulate_sum, "max": accumulate_max}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: str,
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = ACCUMLATORS[accumulator]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.phi(x)  # x.shape = (set_size, input_dim)
        x = self.accumulator(x)
        return self.rho(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)
