from typing import Callable

import torch
import torch.nn as nn

from .config import Model as ModelConfig


def accumulate_sum(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.sum(axis=1)


def accumulate_max(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.max(dim=1).values


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ACCUMLATORS = {"sum": accumulate_sum, "max": accumulate_max}


def generate_model(config: ModelConfig) -> nn.Module:
    model = None
    if config.type == "deepsets_mlp":
        model = DeepSetsInvariant(
            phi=MLP(
                input_dim=config.data_dim,
                hidden_dim=10,
                output_dim=config.laten_dim,
            ),
            rho=MLP(
                input_dim=config.laten_dim,
                hidden_dim=10,
                output_dim=config.data_dim,
            ),
            accumulator=ACCUMLATORS[config.accumulator],
        )
    return model


class DeepSetsInvariant(nn.Module):
    def __init__(
        self,
        phi: nn.Module,
        rho: nn.Module,
        accumulator: Callable[[torch.FloatTensor], torch.FloatTensor],
    ):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.accumulator = accumulator

    def forward(
        self, x: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        x = self.phi(x)  # x.shape = (batch_size, max_set_size, input_dim)
        x = x * mask
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
